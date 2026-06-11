"""
FastAPI inference service for crop disease detection.

Works in two deployment modes:
  - Local Docker:  uvicorn api.main:app --host 0.0.0.0 --port 8000
  - AWS Lambda:    handler = Mangum(app) — CMD ["api.main.handler"]

Endpoints:
  GET  /health   → service status + model info
  POST /predict  → multipart image upload → disease class + confidence

Monitoring:
  Set MONITORING_DB_URL=postgresql://user:pass@host/dbname to enable
  prediction logging to PostgreSQL for Evidently drift monitoring.
"""

import os
import logging
from contextlib import asynccontextmanager

import numpy as np
import psycopg2
from mangum import Mangum  # pylint: disable=import-error
from fastapi import File, FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from api.predict import ModelService, init

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
)
logger = logging.getLogger(__name__)

# module-level singleton — loaded once at cold start, reused across requests
MODEL_SERVICE: ModelService | None = None

# monitoring DB connection — optional, enabled by MONITORING_DB_URL env var
MONITORING_DB_URL = os.getenv("MONITORING_DB_URL")


def _log_prediction_to_db(result: dict) -> None:
    """
    Write prediction to PostgreSQL for drift monitoring.
    Silent no-op if MONITORING_DB_URL is not set or DB is unreachable.
    """
    if not MONITORING_DB_URL:
        return
    try:
        # compute entropy from top_k probabilities
        probs = np.array([item["confidence"] for item in result["top_k"]])
        probs = probs / probs.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-9)))

        conn = psycopg2.connect(MONITORING_DB_URL)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions
                    (predicted_class, confidence, entropy, run_id)
                VALUES (%s, %s, %s, %s)
            """,
                (
                    result["class_name"],
                    result["confidence"],
                    entropy,
                    result["run_id"],
                ),
            )
        conn.commit()
        conn.close()
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Prediction logging failed (non-fatal): %s", exc)


def _ensure_predictions_table() -> None:
    """Create predictions table if it doesn't exist."""
    if not MONITORING_DB_URL:
        return
    try:
        conn = psycopg2.connect(MONITORING_DB_URL)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id             SERIAL PRIMARY KEY,
                        timestamp      TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        predicted_class VARCHAR(100),
                        confidence     FLOAT,
                        entropy        FLOAT,
                        run_id         VARCHAR(100)
                    )
                """
                )
            conn.commit()
        finally:
            conn.close()
        logger.info("Predictions table ready")
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Could not create predictions table: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):  # pylint: disable=redefined-outer-name
    """Load the model at startup."""
    global MODEL_SERVICE  # pylint: disable=global-statement
    logger.info("Loading model...")
    MODEL_SERVICE = init()
    logger.info("Model ready.")
    yield
    # shutdown: nothing to clean up


app = FastAPI(
    title="Crop Disease Detection API",
    description="EfficientNet-B0 fine-tuned on PlantVillage (38 disease classes)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """Return service status and loaded model info."""
    if MODEL_SERVICE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model": "efficientnet_b0",
        "num_classes": len(MODEL_SERVICE.class_names),
        "run_id": MODEL_SERVICE.run_id,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Classify a plant leaf image.

    Accepts a JPEG/PNG image as multipart form data.
    Returns the predicted disease class with confidence score.

    Example:
        curl -X POST http://localhost:8000/predict \\
             -F "file=@leaf.jpg"
    """
    if MODEL_SERVICE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Empty file received")

    result = MODEL_SERVICE.predict(image_bytes)
    _ensure_predictions_table()
    _log_prediction_to_db(result)
    return JSONResponse(content=result)


# Lambda handler — Mangum wraps FastAPI as an ASGI handler
# AWS API Gateway events are converted to ASGI requests transparently
handler = Mangum(app, lifespan="on")

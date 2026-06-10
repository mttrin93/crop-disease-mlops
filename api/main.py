"""
FastAPI inference service for crop disease detection.

Works in two deployment modes:
  - Local Docker:  uvicorn api.main:app --host 0.0.0.0 --port 8000
  - AWS Lambda:    handler = Mangum(app) — CMD ["api.main.handler"]

Endpoints:
  GET  /health   → service status + model info
  POST /predict  → multipart image upload → disease class + confidence
"""

import logging
from contextlib import asynccontextmanager

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
    return JSONResponse(content=result)


# Lambda handler — Mangum wraps FastAPI as an ASGI handler
# AWS API Gateway events are converted to ASGI requests transparently
handler = Mangum(app, lifespan="on")

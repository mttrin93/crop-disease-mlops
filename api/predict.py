"""
Model loading and inference logic for the crop disease detection service.

Loads an ONNX model exported from the training notebook and runs inference
on incoming images. Separated from main.py so it can be tested independently
without starting the FastAPI server.

RUN_ID resolution (in priority order):
  1. RUN_ID env var          → set directly (local testing, Lambda env var)
  2. SSM Parameter Store     → /crop-disease-mlops/staging/run_id (production)

Model source resolution (in priority order):
  1. MODEL_LOCATION env var  → local path (local Docker testing)
  2. MLflow tracking server  → downloads ONNX artifact via RUN_ID from S3
"""

import io
import os
import json
import logging
import tempfile
from pathlib import Path

import boto3
import numpy as np
import mlflow
import onnxruntime as ort
from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet normalisation — must match training transforms
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 224

# SSM parameter name where the CD pipeline writes the active RUN_ID
SSM_PARAMETER_NAME = os.getenv(
    "SSM_PARAMETER_NAME",
    "/crop-disease-mlops/staging/run_id",
)


# ---------- RUN_ID resolution ----------


def get_run_id() -> str:
    """
    Resolve the active MLflow RUN_ID.

    Priority:
      1. RUN_ID env var  — set directly for local testing or by CD pipeline
                           as a Lambda environment variable
      2. SSM Parameter Store — production path: the CD pipeline writes the
                           promoted RUN_ID here after training
    """
    run_id = os.getenv("RUN_ID")
    if run_id:
        logger.info("RUN_ID from env var: %s", run_id)
        return run_id

    logger.info("RUN_ID not in env, reading from SSM...")
    ssm_parameter_name = os.getenv(
        "SSM_PARAMETER_NAME",
        "/crop-disease-mlops/staging/run_id",
    )
    ssm_client = boto3.client(
        "ssm",
        region_name=os.getenv("AWS_DEFAULT_REGION", "eu-west-1"),
    )
    response = ssm_client.get_parameter(Name=ssm_parameter_name)
    run_id = response["Parameter"]["Value"]
    logger.info("RUN_ID from SSM (%s): %s", ssm_parameter_name, run_id)
    return run_id


# ---------- preprocessing ----------


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes → normalised NCHW float32 tensor.

    Replicates the eval_transforms from the training notebook:
      Resize → ToTensor → Normalize(ImageNet mean/std)

    Returns shape (1, 3, 224, 224) — batch size 1.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    # HWC → CHW → NCHW
    return arr.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)


# ---------- model loading ----------


def _download_from_mlflow(run_id: str, dst_dir: str) -> str:
    """Download ONNX model and metadata from MLflow artifact store."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        raise ValueError("MLFLOW_TRACKING_URI env var not set")

    mlflow.set_tracking_uri(mlflow_uri)
    logger.info("Downloading artifacts for run %s from MLflow...", run_id)
    logger.info("destination folder %s", dst_dir)

    # local_path = mlflow.artifacts.download_artifacts(
    #    run_id=run_id,
    #    artifact_path="onnx",
    #        artifact_path="models",
    #    dst_path=dst_dir,
    # )

    client = mlflow.MlflowClient()
    local_path = client.download_artifacts(run_id, "onnx", dst_dir)

    return local_path


def load_model_and_metadata() -> tuple[ort.InferenceSession, dict, str]:
    """
    Load ONNX model and class metadata.

    Returns:
        session:    ONNX Runtime inference session
        metadata:   dict with class_names, class_to_index, etc.
        run_id:     MLflow run ID of the loaded model
    """
    run_id = get_run_id()
    model_location = os.getenv("MODEL_LOCATION")

    if model_location:
        # local path override — used for local Docker testing
        artifact_dir = model_location
        logger.info("Loading model from local path: %s", artifact_dir)
    else:
        # download from MLflow artifact store
        tmp_dir = tempfile.mkdtemp(prefix="crop_disease_model_")
        artifact_dir = _download_from_mlflow(run_id, tmp_dir)

    model_path = Path(artifact_dir) / "model.onnx"
    metadata_path = Path(artifact_dir) / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"model.onnx not found at {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {metadata_path}")

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(
        "Model loaded: %s classes, run_id=%s",
        len(metadata["class_names"]),
        run_id,
    )
    return session, metadata, run_id


# ---------- inference ----------


class ModelService:  # pylint: disable=too-few-public-methods
    """
    Wraps ONNX Runtime session and exposes a simple predict() interface.
    Instantiated once at application startup and reused across requests.
    """

    def __init__(self, session: ort.InferenceSession, metadata: dict, run_id: str):
        self.session = session
        self.metadata = metadata
        self.run_id = run_id
        self.class_names: list[str] = metadata["class_names"]
        self.input_name: str = session.get_inputs()[0].name

    def predict(self, image_bytes: bytes, top_k: int = 5) -> dict:
        """
        Run inference on raw image bytes.

        Returns:
            class_name:   predicted disease class
            confidence:   probability of the top prediction (0–1)
            top_k:        list of {class, confidence} for the top-k predictions
            run_id:       MLflow run ID of the model
        """
        tensor = preprocess_image(image_bytes)
        logits = self.session.run(None, {self.input_name: tensor})[0][0]

        # softmax
        exp_logits = np.exp(logits - logits.max())
        probabilities = exp_logits / exp_logits.sum()

        top_indices = probabilities.argsort()[::-1][:top_k]

        return {
            "class_name": self.class_names[top_indices[0]],
            "confidence": float(probabilities[top_indices[0]]),
            "top_k": [
                {
                    "class_name": self.class_names[i],
                    "confidence": float(probabilities[i]),
                }
                for i in top_indices
            ],
            "run_id": self.run_id,
        }


def init(run_id: str | None = None) -> ModelService:
    """Convenience factory — loads model and returns a ModelService."""
    if run_id:
        os.environ["RUN_ID"] = run_id
    session, metadata, loaded_run_id = load_model_and_metadata()
    return ModelService(session, metadata, loaded_run_id)

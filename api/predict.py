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


def _download_from_s3(run_id: str, dst_dir: str) -> str:
    """
    Download full ONNX artifact folder from S3 (MLflow-style export).
    """
    bucket = os.getenv("MODEL_BUCKET")
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", "1")
    prefix = f"{experiment_id}/{run_id}/artifacts/onnx"

    s3 = boto3.client("s3")

    local_dir = Path(dst_dir) / "onnx"
    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading s3://%s -> %s", prefix, local_dir)

    paginator = s3.get_paginator("list_objects_v2")
    found = False

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            found = True
            key = obj["Key"]

            relative_path = Path(key).relative_to(prefix)
            target_path = local_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)

            s3.download_file(bucket, key, str(target_path))

            logger.debug("Downloaded %s", key)

    if not found:
        raise FileNotFoundError(f"No artifacts found at s3://{bucket}/{prefix}")

    return str(local_dir)


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
        # download from s3 artifact store
        tmp_dir = tempfile.mkdtemp(prefix="crop_disease_model_")
        artifact_dir = _download_from_s3(run_id, tmp_dir)

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

# Inference Service — Local Testing Guide

This guide covers how to build and test the crop disease inference service
locally before deploying to AWS Lambda.

---

## Prerequisites

- Docker installed and running
- AWS credentials configured (`aws configure`)
- MLflow server running on EC2 (see `notebooks/README.md`)
- A trained model registered in MLflow with a known `RUN_ID`
- Processed test images at `data/processed/test/`

---

## Build the image

From the project root:

```bash
docker build -t crop-disease-inference:latest -f api/Dockerfile .
```

> Make sure `.dockerignore` is in the project root to avoid sending the
> 4GB `data/` folder to the build context.

---

## Option A — FastAPI mode (recommended for local testing)

Overrides the Lambda entrypoint with uvicorn — pure FastAPI, no Lambda
event format needed. Simpler to test with standard HTTP tools.

```bash
docker run -p 8000:8000 \
    --entrypoint uvicorn \
    -v ~/.aws:/root/.aws:ro \
    -e RUN_ID=31c978e10c9e440c8a82f882b6b65e9f \
    -e MLFLOW_TRACKING_URI=http://your-ec2-dns:5000 \
    -e AWS_DEFAULT_REGION=eu-west-1 \
    crop-disease-inference:latest \
    api.main:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
# health check
curl http://localhost:8000/health

# predict with a local test image
curl -X POST http://localhost:8000/predict \
    -F "file=@data/processed/test/Tomato___healthy/000000.jpg"
```

Expected response:

```json
{
  "class_name": "Tomato___healthy",
  "confidence": 0.97,
  "top_k": [
    {"class_name": "Tomato___healthy", "confidence": 0.97},
    {"class_name": "Tomato___Late_blight", "confidence": 0.02},
    ...
  ],
  "run_id": "31c978e10c9e440c8a82f882b6b65e9f"
}
```

---

## Option B — Lambda RIE mode (closer to production)

Uses the Lambda Runtime Interface Emulator (RIE) built into the base image.
Tests the actual Lambda handler and Mangum adapter. Requires the Lambda
event format.

```bash
# health check
python tests/integration/test_lambda_local.py --health

# predict
python tests/integration/test_lambda_local.py \
    --image data/processed/test/Tomato___healthy/000004.jpg
```

## Environment variables reference

| Variable | Required | Description |
|---|---|---|
| `RUN_ID` | Yes (or SSM) | MLflow run ID of the trained model |
| `MLFLOW_TRACKING_URI` | Yes | URL of the MLflow server on EC2 |
| `AWS_DEFAULT_REGION` | Yes | AWS region (e.g. `eu-west-1`) |
| `SSM_PARAMETER_NAME` | No | SSM path for RUN_ID (default: `/crop-disease-mlops/staging/run_id`) |
| `MODEL_LOCATION` | No | Local path override — skips MLflow download entirely |

If `RUN_ID` is not set as an env var, the service reads it automatically
from SSM Parameter Store at the path defined by `SSM_PARAMETER_NAME`.

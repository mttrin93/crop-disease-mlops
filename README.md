# Crop Disease Detection — End-to-End MLOps Pipeline

> **An end-to-end MLOps system for automated crop disease detection using computer vision.**
> Built as a portfolio project applying all modules of the DataTalksClub MLOps Zoomcamp 2025.

[![CI Tests](https://github.com/mttrin93/crop-disease-mlops/actions/workflows/ci.yaml/badge.svg)](https://github.com/mttrin93/crop-disease-mlops/actions/workflows/ci.yaml)
[![CI Terraform](https://github.com/mttrin93/crop-disease-mlops/actions/workflows/ci-tf.yaml/badge.svg)](https://github.com/mttrin93/crop-disease-mlops/actions/workflows/ci-tf.yaml)
[![CD Deploy](https://github.com/mttrin93/crop-disease-mlops/actions/workflows/cd.yaml/badge.svg?branch=develop)](https://github.com/mttrin93/crop-disease-mlops/actions/workflows/cd.yaml)

---

## Problem Description

Crop diseases are responsible for significant agricultural yield losses worldwide.
Early and accurate detection is critical for farmers to take timely action.
This project builds a production-ready MLOps pipeline that:

- Classifies plant leaf images into **38 disease categories** across multiple crops
- Serves predictions via a **REST API** deployed on AWS Lambda
- Monitors **prediction drift** in production with Evidently + Grafana
- Retrains automatically on a weekly schedule via **Apache Airflow**

---

## Architecture

```
PlantVillage Dataset (downloaded locally)
         │
         ▼
src/data/preprocess.py
  └── leaf-aware train/val/test split (70/15/15)
  └── resize to 224×224 → data/processed/
         │
         ▼ (manual upload)
S3 bucket — data + model artifacts
         │
         ▼
Google Colab — notebooks/train_colab.ipynb
  └── EfficientNet-B0 fine-tuning (PyTorch)
  └── MLflow: track metrics, log ONNX model
  └── register in MLflow Model Registry
  └── generate reference data for monitoring
         │
         ▼
MLflow Model Registry (EC2)
  └── Staging → Production promotion
         │
         ├─────────────────────────────────────┐
         ▼                                     ▼
FastAPI (local Docker)              AWS Lambda + API Gateway
  └── development / testing           └── production inference
         │
         ▼
Evidently drift monitoring
  └── reference: test-set predictions (S3)
  └── current: API predictions (PostgreSQL)
  └── Grafana dashboard + conditional retraining

Orchestration:  Airflow (training pipeline DAG + monitoring DAG)
Infrastructure: Terraform (S3, ECR, Lambda, API Gateway, IAM)
CI/CD:          GitHub Actions (3 workflows)
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Model | EfficientNet-B0 (PyTorch → ONNX) |
| Experiment tracking | MLflow (EC2 + RDS PostgreSQL + S3) |
| Model registry | MLflow Model Registry |
| Orchestration | Apache Airflow (Docker Compose) |
| Serving — dev | FastAPI + uvicorn + Docker |
| Serving — prod | FastAPI + Mangum + AWS Lambda + API Gateway |
| Monitoring | Evidently + Grafana + PostgreSQL |
| Infrastructure | Terraform (AWS: S3, ECR, Lambda, API Gateway, IAM) |
| CI/CD | GitHub Actions (3 workflows) |
| Dependency management | uv |

---

## Dataset

**PlantVillage** — `TSY-0408/PlantVillage` on HuggingFace

- ~54,000 images across 38 disease classes
- 14 crop species (tomato, potato, apple, corn, grape, and more)
- **Leaf-aware splitting**: images of the same physical leaf are never split across train and test — prevents data leakage
- License: CC BY 4.0

---

## Project Structure

```
crop-disease-mlops/
├── src/
│   ├── data/
│   │   └── preprocess.py         # leaf-aware split, resize, save to data/processed/
│   ├── model/
│   │   └── train.py              # EfficientNet-B0, MLflow logging, ONNX export
│   └── monitoring/
│       └── drift.py              # Evidently drift report → PostgreSQL
├── api/
│   ├── main.py                   # FastAPI app + Mangum Lambda handler
│   ├── predict.py                # ONNX inference, RUN_ID from env/SSM
│   ├── pyproject.toml            # api-only dependencies (no mlflow, no torch)
│   └── Dockerfile                # Lambda container image (~600 MB)
├── airflow/
│   ├── docker-compose.yaml       # Airflow stack (LocalExecutor + PostgreSQL)
│   ├── README.md                 # setup guide
│   └── dags/
│       ├── training_pipeline.py  # weekly: preprocess → train → evaluate → deploy
│       ├── monitoring_dag.py     # daily: drift check → conditional retrain
│       └── dag_tasks.py          # pure Python callables (no Airflow imports, fully testable)
├── monitoring/
│   ├── docker-compose.yaml       # Grafana + PostgreSQL
│   ├── README.md                 # setup guide
│   ├── config/
│   │   ├── grafana_datasources.yaml
│   │   └── grafana_dashboards.yaml
│   └── dashboards/
│       └── drift_dashboard.json  # 8-panel Grafana dashboard
├── infrastructure/
│   ├── main.tf                   # S3, ECR, Lambda modules
│   ├── variables.tf
│   ├── outputs.tf
│   ├── terraform.tfvars.example
│   ├── README.md                 # setup guide + known issues
│   └── modules/
│       ├── s3/                   # model artifact bucket
│       ├── ecr/                  # Docker image registry + bootstrap push
│       └── lambda/               # Lambda + API Gateway v2 + IAM roles
├── tests/
│   ├── unit/
│   │   ├── test_preprocess.py    # leaf-aware split, image processing
│   │   ├── test_predict.py       # ONNX inference, SSM RUN_ID resolution
│   │   ├── test_drift.py         # Evidently metrics, PostgreSQL writes
│   │   └── test_training_dag.py  # Airflow callable functions
│   └── integration/
│       ├── test_api.py           # HTTP tests against live or local API
│       ├── run.sh                # local Docker integration test runner
│       └── test_lambda_local.py  # Lambda RIE local test (multipart events)
├── notebooks/
│   ├── train_colab.ipynb         # full training + ONNX export + reference data
│   └── README.md                 # training setup guide (EC2, MLflow, S3)
├── .github/workflows/
│   ├── ci.yaml                   # PR: lint (black, isort, pylint) + unit tests
│   ├── ci-tf.yaml                # PR touching infrastructure/: terraform plan
│   └── cd.yaml                   # push to develop: build → ECR → Lambda → integration test
├── .pre-commit-config.yaml
├── Makefile                      # all common tasks (make help for list)
├── pyproject.toml                # dev + training + monitoring dependency groups
└── .env.example
```

---

## Quick Start

### Prerequisites

- Python 3.11, Docker, Terraform >= 1.0
- AWS CLI configured (`aws configure`)
- uv installed (`pip install uv`)

### 1. Clone and set up

```bash
git clone https://github.com/mttrin93/crop-disease-mlops.git
cd crop-disease-mlops
make install
```

### 2. Configure environment

```bash
cp .env.example .env
# fill in your AWS credentials and MLflow URI
```

### 3. Download data and preprocess

Download the PlantVillage dataset manually and unzip under `data/raw/color/`:
```
data/raw/color/
    Apple___Apple_scab/    *.JPG
    Apple___Black_rot/     *.JPG
    ...                    (38 class directories)
```

Then run preprocessing:
```bash
python src/data/preprocess.py
# → data/processed/train/, val/, test/ + metadata.json
```

### 4. Upload processed data to S3

```bash
make upload-processed
```

### 5. Train the model (Google Colab)

Open `notebooks/train_colab.ipynb` in Colab (GPU runtime).
See `notebooks/README.md` for full setup instructions.

The notebook:
- Downloads processed data from S3
- Fine-tunes EfficientNet-B0
- Logs metrics and ONNX model to MLflow
- Registers the model in MLflow Model Registry
- Generates reference data for drift monitoring → S3

### 6. Run the inference service locally

```bash
make build
make run
# set RUN_ID and MLFLOW_TRACKING_URI in .env first

# test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@leaf.jpg"
```

### 7. Provision AWS infrastructure

```bash
cd infrastructure
cp terraform.tfvars.example terraform.tfvars
# fill in run_id and mlflow_tracking_uri

terraform init
terraform import module.s3_bucket.aws_s3_bucket.main YOUR_BUCKET_NAME
terraform plan
terraform apply
```

See `infrastructure/README.md` for full details and known issues.

### 8. Start monitoring

```bash
make monitoring-up
# Grafana at http://localhost:3000 (admin/admin)
# run drift monitoring
make drift
```

### 9. Start Airflow

```bash
make airflow-init   # first time only
make airflow-up
# UI at http://localhost:8081 (admin/admin)
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status + model info |
| `POST` | `/predict` | Upload leaf image → disease class |

### Example request

```bash
curl -X POST https://your-api-gateway-url/predict \
  -F "file=@leaf.jpg"
```

### Example response

```json
{
  "class_name": "Tomato___Late_blight",
  "confidence": 0.94,
  "top_k": [
    {"class_name": "Tomato___Late_blight", "confidence": 0.94},
    {"class_name": "Tomato___healthy",     "confidence": 0.04},
    ...
  ],
  "run_id": "afe0821a13c74ed2a10429da27ac4577"
}
```

---

## Monitoring

The monitoring stack tracks production predictions against a reference
distribution computed at training time (test-set predictions).

| Metric | Threshold | Action |
|---|---|---|
| `share_of_drifted_columns` | > 0.15 | trigger retraining |
| `num_drifted_columns` | > 5 | trigger retraining |
| `share_missing_values` | > 0.05 | trigger retraining |

Reference data is generated in Cell 10 of `train_colab.ipynb` after training.
The Airflow `monitoring_dag` runs daily and automatically triggers
`training_pipeline` if thresholds are exceeded.

See `monitoring/README.md` for setup instructions.

---

## CI/CD

| Workflow | Trigger | Jobs |
|---|---|---|
| `ci.yaml` | PR to main/develop (src, api, tests changed) | lint (black, isort, pylint) + unit tests |
| `ci-tf.yaml` | PR to main/develop (infrastructure/ changed) | terraform validate + plan |
| `cd.yaml` | Push to develop (api/ changed) | build → ECR → Lambda update → integration tests |

The CD pipeline:
1. Builds and pushes Docker image tagged with git SHA
2. Updates Lambda function code
3. Fetches active `RUN_ID` from SSM Parameter Store
4. Updates Lambda environment variables
5. Runs integration tests against the live API endpoint

---

## Cost Management

All AWS resources are provisioned with Terraform and can be cleanly torn down:

```bash
# destroy Lambda + API Gateway only (keep S3 + ECR)
cd infrastructure && terraform destroy -target=module.lambda

# stop EC2 MLflow server (preserves setup)
aws ec2 stop-instances --instance-ids YOUR_INSTANCE_ID --region eu-west-1

# stop RDS (preserves data)
aws rds stop-db-instance --db-instance-identifier YOUR_DB_ID --region eu-west-1
```

Approximate cost for an active session (2-3 hours of training + testing): **< $0.50**

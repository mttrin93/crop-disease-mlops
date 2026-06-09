# Crop Disease Detection — End-to-End MLOps Pipeline

> **An end-to-end MLOps system for automated crop disease detection using computer vision.**
> Built as a portfolio project applying all modules of the DataTalksClub MLOps Zoomcamp 2025.

[![CI Tests](https://github.com/mttrin93/crop-disease-mlops/actions/workflows/ci.yaml/badge.svg)](https://github.com/mttrin93/crop-disease-mlops/actions/workflows/ci.yaml)
[![CD Deploy](https://github.com/mttrin93/crop-disease-mlops/actions/workflows/cd.yaml/badge.svg)](https://github.com/mttrin93/crop-disease-mlops/actions/workflows/cd.yaml)

---

## Problem Description

Crop diseases are responsible for significant agricultural yield losses worldwide.
Early and accurate detection is critical for farmers to take timely action.
This project builds a production-ready MLOps pipeline that:

- Classifies plant leaf images into **38 disease categories** across multiple crops
- Serves predictions via a **REST API** deployed on AWS Lambda
- Monitors **model drift and data quality** in production via Evidently + Grafana
- Retrains automatically on a weekly schedule via **Apache Airflow**

The downstream output feeds into a commodity price impact model (Part 2 of this project).

---

## Architecture

```
PlantVillage Dataset (HuggingFace)
         │
         ▼
Airflow DAG — weekly orchestration
  ├── Task 1: download + preprocess images → S3
  ├── Task 2: trigger Colab training notification
  └── Task 3: register best model in MLflow Registry
         │
         ▼
EfficientNet-B0 Training (Google Colab + PyTorch)
  └── MLflow: track metrics, log ONNX model, register
         │
         ▼
MLflow Model Registry
  └── staging → production promotion
         │
         ├── FastAPI (local Docker) ← development
         └── AWS Lambda + API Gateway ← production
                  │
                  ▼
         Evidently monitoring
         └── Grafana dashboard + drift alerts
```

Infrastructure: **Terraform** (all AWS resources as code)
CI/CD: **GitHub Actions** (CI on PR, CD on merge to develop)

---

## Tech Stack

| Component | Tool |
|---|---|
| Model | EfficientNet-B0 (PyTorch → ONNX) |
| Experiment tracking | MLflow |
| Orchestration | Apache Airflow |
| Serving (dev) | FastAPI + Docker |
| Serving (prod) | AWS Lambda + API Gateway |
| Monitoring | Evidently + Grafana + PostgreSQL |
| Infrastructure | Terraform (AWS: S3, EC2, ECR, Lambda) |
| CI/CD | GitHub Actions |
| Dependency management | uv |

---

## Dataset

**PlantVillage** — `TSY-0408/PlantVillage` on HuggingFace

- 87,000 images across 38 disease classes
- 14 crop species (tomato, potato, apple, corn, grape, and more)
- Balanced classes, clean labels, widely cited in academic literature
- License: CC BY 4.0

---

## Project Structure

```
crop-disease-mlops/
├── src/
│   ├── data/
│   │   └── preprocess.py        # resize, augment, train/val/test split → S3
│   ├── model/
│   │   ├── train.py             # EfficientNet-B0 fine-tuning + MLflow logging
│   │   └── evaluate.py          # F1, accuracy, confusion matrix
│   └── monitoring/
│       └── drift.py             # Evidently drift report + PostgreSQL write
├── api/
│   ├── main.py                  # FastAPI app (health, predict endpoints)
│   ├── predict.py               # ONNX model loading + inference
│   └── Dockerfile               # works for local Docker and AWS Lambda
├── airflow/
│   ├── docker-compose.yaml      # local Airflow stack
│   └── dags/
│       └── training_pipeline.py # weekly DAG: ingest → train → register
├── monitoring/
│   ├── docker-compose.yaml      # Grafana + PostgreSQL
│   ├── batch_monitor.py         # compute Evidently metrics → PostgreSQL
│   └── config/
│       ├── grafana_datasources.yaml
│       └── grafana_dashboards.yaml
├── infrastructure/
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── modules/
│       ├── s3/                  # data + model artifact bucket
│       ├── ecr/                 # Docker image registry
│       ├── lambda/              # inference function + API Gateway
│       └── ec2/                 # MLflow tracking server
├── tests/
│   ├── unit/                    # unit tests for preprocessing + inference
│   └── integration/             # integration test against Lambda container
├── notebooks/
│   └── train_colab.ipynb        # full training notebook (run on Google Colab)
├── .github/workflows/
│   ├── ci.yaml                  # on PR: quality checks + unit tests
│   └── cd.yaml                  # on merge: build + deploy to AWS
├── Makefile
├── pyproject.toml
└── .env.example
```

---

## Quick Start

### Prerequisites

- Python 3.11
- Docker
- AWS CLI configured (`aws configure`)
- Terraform >= 1.0

### 1. Clone and set up

```bash
git clone https://github.com/mttrin93/crop-disease-mlops.git
cd crop-disease-mlops
make setup
```

### 2. Configure environment

```bash
cp .env.example .env
# fill in your AWS credentials and MLflow URI
```

### 3. Download data and preprocess

```bash
uv run python src/data/download.py
uv run python src/data/preprocess.py
```

### 4. Train the model (Google Colab)

Open `notebooks/train_colab.ipynb` in Google Colab and run all cells.
The notebook will:
- Load preprocessed images from S3
- Fine-tune EfficientNet-B0
- Log metrics and model to MLflow
- Export to ONNX and register in MLflow Model Registry

### 5. Run the inference service locally

```bash
make build
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://your-ec2:5000 \
  -e RUN_ID=your_run_id \
  crop-disease-inference:latest

# test it
curl -X POST http://localhost:8000/predict \
  -F "file=@test_leaf.jpg"
```

### 6. Start monitoring stack

```bash
make monitoring
# open http://localhost:3000 (Grafana)
```

### 7. Provision AWS infrastructure

```bash
cd infrastructure
cp terraform.tfvars.example terraform.tfvars
# fill in terraform.tfvars
terraform init
terraform plan
terraform apply
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check + model version |
| `POST` | `/predict` | Upload leaf image → disease class + confidence |

### Example request

```bash
curl -X POST https://your-api-gateway-url/predict \
  -F "file=@leaf.jpg"
```

### Example response

```json
{
  "class": "Tomato___Late_blight",
  "confidence": 0.94,
  "run_id": "abc123def456",
  "model_version": "3"
}
```

---

## Monitoring

The monitoring stack tracks:

- **Prediction confidence distribution** — detects when the model is uncertain
- **Class distribution drift** — detects shift in incoming image types
- **Data quality** — missing values, image shape anomalies
- **Model performance** — F1 score on labeled production samples (when available)

Access the Grafana dashboard at `http://localhost:3000` (local) or your EC2 Grafana instance.

---

## Evaluation Criteria Coverage

| Criterion | Implementation | Score |
|---|---|---|
| Problem description | This README + architecture diagram | 2/2 |
| Cloud + IaC | AWS (S3, EC2, ECR, Lambda) + Terraform | 4/4 |
| Experiment tracking + registry | MLflow tracking + Model Registry | 4/4 |
| Workflow orchestration | Airflow DAG (fully deployed) | 4/4 |
| Model deployment | Dockerized FastAPI → AWS Lambda | 4/4 |
| Model monitoring | Evidently + Grafana + conditional alerts | 4/4 |
| Reproducibility | make setup, pinned deps, data download script | 4/4 |
| Best practices | unit tests, integration test, black+isort+pylint, Makefile, pre-commit, CI/CD | 7/7 |

---

## Author

Matteo Rinaldi — [GitHub](https://github.com/mttrin93) · [LinkedIn](https://linkedin.com/in/matteo-rinaldi)

PhD in Computational Physics · ML Engineer · Berlin

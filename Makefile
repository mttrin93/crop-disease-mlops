# ──────────────────────────────────────────────────────────────────────────────
# Crop Disease Detection MLOps — Makefile
# ──────────────────────────────────────────────────────────────────────────────

# load .env if it exists
-include .env
export

# ── variables ─────────────────────────────────────────────────────────────────

AWS_REGION        ?= eu-west-1
MODEL_BUCKET      ?= crop-disease-models-mlflow-stg-478544568263
ECR_REPO_URL      ?= 478544568263.dkr.ecr.eu-west-1.amazonaws.com/crop-disease-inference_crop-disease-mlops
LAMBDA_FUNCTION   ?= crop-disease-predict_crop-disease-mlops
API_URL           ?= https://bf17kc4e1b.execute-api.eu-west-1.amazonaws.com
IMAGE_TAG         ?= latest

# ── development ───────────────────────────────────────────────────────────────

.PHONY: install
install:  ## Install all development dependencies
	uv sync --group api --group monitoring --group dev
	uv pip install scikit-learn pillow mlflow

.PHONY: lint
lint:  ## Run black, isort, pylint
	uv run black src/ api/ tests/
	uv run isort src/ api/ tests/
	uv run pylint src/ api/ --fail-under=8.0

.PHONY: test
test:  ## Run unit tests
	uv run pytest tests/unit/ -v --tb=short

.PHONY: test-all
test-all:  ## Run all tests including integration
	uv run pytest tests/ -v --tb=short

# ── data pipeline ─────────────────────────────────────────────────────────────

.PHONY: preprocess
preprocess:  ## Preprocess raw images → data/processed/
	uv run python src/data/preprocess.py

.PHONY: upload-processed
upload-processed:  ## Upload processed data to S3
	aws s3 sync data/processed/ s3://$(MODEL_BUCKET)/data/processed/ \
		--region $(AWS_REGION)

# ── docker ────────────────────────────────────────────────────────────────────

.PHONY: build
build:  ## Build the inference Docker image
	docker build -t crop-disease-inference:$(IMAGE_TAG) -f api/Dockerfile .

.PHONY: run
run:  ## Run the inference service locally (uvicorn mode)
	docker run -p 8000:8000 \
		--entrypoint uvicorn \
		-v ~/.aws:/root/.aws:ro \
		-e RUN_ID=$(RUN_ID) \
		-e MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
		--network host \
		crop-disease-inference:$(IMAGE_TAG) \
		api.main:app --host 0.0.0.0 --port 8000

.PHONY: run-lambda
run-lambda:  ## Run the inference service locally (Lambda RIE mode)
	docker run -p 8080:8080 \
		-v ~/.aws:/root/.aws:ro \
		-e RUN_ID=$(RUN_ID) \
		-e MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
		crop-disease-inference:$(IMAGE_TAG)

# ── ECR ───────────────────────────────────────────────────────────────────────

.PHONY: ecr-login
ecr-login:  ## Login to ECR
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin \
		$(shell echo $(ECR_REPO_URL) | cut -d/ -f1)

.PHONY: push
push: ecr-login  ## Build and push image to ECR
	docker build -t $(ECR_REPO_URL):$(IMAGE_TAG) -f api/Dockerfile .
	docker push $(ECR_REPO_URL):$(IMAGE_TAG)

# ── Lambda ────────────────────────────────────────────────────────────────────

.PHONY: deploy
deploy:  ## Update Lambda with latest ECR image
	aws lambda update-function-code \
		--function-name $(LAMBDA_FUNCTION) \
		--image-uri $(ECR_REPO_URL):$(IMAGE_TAG) \
		--region $(AWS_REGION)
	aws lambda wait function-updated \
		--function-name $(LAMBDA_FUNCTION) \
		--region $(AWS_REGION)
	@echo "Lambda updated successfully"

.PHONY: update-run-id
update-run-id:  ## Update Lambda RUN_ID env var (make update-run-id RUN_ID=abc123)
	aws lambda update-function-configuration \
		--function-name $(LAMBDA_FUNCTION) \
		--environment "Variables={RUN_ID=$(RUN_ID),MODEL_BUCKET=$(MODEL_BUCKET),MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI),MLFLOW_EXPERIMENT_ID=1}" \
		--region $(AWS_REGION)

.PHONY: logs
logs:  ## Tail Lambda CloudWatch logs
	aws logs tail /aws/lambda/$(LAMBDA_FUNCTION) \
		--follow --region $(AWS_REGION)

# ── smoke tests ───────────────────────────────────────────────────────────────

.PHONY: smoke-test
smoke-test:  ## Test the deployed Lambda API
	@echo "Testing $(API_URL)/health"
	@curl -sf $(API_URL)/health | python3 -m json.tool
	@echo "\nTesting $(API_URL)/predict"
	@curl -sf -X POST $(API_URL)/predict \
		-F "file=@data/processed/test/Tomato___healthy/000004.jpg" | python3 -m json.tool

# ── Terraform ─────────────────────────────────────────────────────────────────

.PHONY: tf-plan
tf-plan:  ## Terraform plan
	cd infrastructure && terraform plan

.PHONY: tf-apply
tf-apply:  ## Terraform apply
	cd infrastructure && terraform apply

.PHONY: tf-destroy
tf-destroy:  ## Destroy Lambda and ECR only (keep S3)
	cd infrastructure && terraform destroy \
		-target=module.lambda \
		-target=module.ecr

# ── monitoring ────────────────────────────────────────────────────────────────

.PHONY: monitoring-up
monitoring-up:  ## Start Grafana + PostgreSQL monitoring stack
	docker compose -f monitoring/docker-compose.yaml up -d

.PHONY: monitoring-down
monitoring-down:  ## Stop monitoring stack
	docker compose -f monitoring/docker-compose.yaml down

.PHONY: drift
drift:  ## Run drift monitoring (quick test with synthetic data)
	uv run python src/monitoring/drift.py --skip-db

# ── airflow ───────────────────────────────────────────────────────────────────

.PHONY: airflow-init
airflow-init:  ## Initialise Airflow (first time only)
	mkdir -p airflow/logs airflow/plugins
	echo "AIRFLOW_UID=$$(id -u)" > airflow/.env
	docker compose -f airflow/docker-compose.yaml up airflow-init

.PHONY: airflow-up
airflow-up:  ## Start Airflow (UI at http://localhost:8081)
	docker compose -f airflow/docker-compose.yaml up

.PHONY: airflow-down
airflow-down:  ## Stop Airflow
	docker compose -f airflow/docker-compose.yaml down

# ── help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

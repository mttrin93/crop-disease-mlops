LOCAL_TAG := $(shell date +"%Y-%m-%d-%H-%M")
LOCAL_IMAGE_NAME := crop-disease-inference:$(LOCAL_TAG)

.PHONY: setup test quality build integration-test monitoring airflow deploy help

help:
	@echo "crop-disease-mlops — available targets:"
	@echo ""
	@echo "  setup              Install dependencies and pre-commit hooks"
	@echo "  test               Run unit tests"
	@echo "  quality            Run isort + black + pylint"
	@echo "  build              Build Docker image for inference service"
	@echo "  integration-test   Run integration tests against local container"
	@echo "  monitoring         Start Grafana + PostgreSQL monitoring stack"
	@echo "  airflow            Start Airflow orchestration stack"
	@echo "  deploy             Deploy to AWS (Terraform + Lambda)"

setup:
	pip install uv
	uv sync --dev
	uv run pre-commit install

test:
	uv run pytest tests/unit/ -v

quality:
	uv run isort src/ api/ tests/
	uv run black src/ api/ tests/
	uv run pylint src/ api/ tests/ --recursive=y

build: quality test
	docker build -t $(LOCAL_IMAGE_NAME) -f api/Dockerfile .

integration-test: build
	LOCAL_IMAGE_NAME=$(LOCAL_IMAGE_NAME) bash tests/integration/run.sh

monitoring:
	docker compose -f monitoring/docker-compose.yaml up

airflow:
	docker compose -f airflow/docker-compose.yaml up

deploy: build integration-test
	LOCAL_IMAGE_NAME=$(LOCAL_IMAGE_NAME) bash scripts/deploy.sh

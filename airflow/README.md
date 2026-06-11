# Airflow Orchestration — Setup Guide

This guide documents how to run the Airflow orchestration stack locally.
Airflow manages two DAGs:

- **`crop_disease_training_pipeline`** — weekly: preprocess → train → evaluate → deploy
- **`crop_disease_monitoring`** — daily: compute drift metrics → trigger retraining if needed

---

## Prerequisites

- Docker and Docker Compose installed
- AWS CLI configured (`aws configure`)
- MLflow server running on EC2 (see `notebooks/README.md`)
- Processed data in S3 (see Step 2)

---

## Directory structure

```
airflow/
├── docker-compose.yaml       ← Airflow stack (scheduler + webserver + postgres)
├── .env                      ← auto-generated, contains AIRFLOW_UID
├── dags/
│   ├── training_pipeline.py  ← weekly training DAG
│   ├── monitoring_dag.py     ← daily monitoring DAG
│   └── dag_tasks.py          ← Python callables (no Airflow dependency)
├── logs/                     ← task execution logs (auto-created)
└── plugins/                  ← custom Airflow plugins (empty for now)
```

---

## First-time setup

### 1. Create required directories

Docker may create `logs/` with root ownership on first run causing permission
errors. Always create it manually before starting:

```bash
sudo rm -rf airflow/logs airflow/plugins   # remove if already created by Docker
mkdir -p airflow/logs airflow/plugins
```

### 2. Create the .env file

Airflow runs as a non-root user inside the container. This sets the container
user ID to match your host user so it can write to mounted directories:

```bash
echo "AIRFLOW_UID=$(id -u)" > airflow/.env
```

### 3. Set environment variables

Copy `.env.example` to `.env` in the project root and fill in your values,
or export them in your shell before starting Airflow:

```bash
export MLFLOW_TRACKING_URI=...
export MODEL_BUCKET=...
export AWS_DEFAULT_REGION=...
export LAMBDA_FUNCTION=...
```

### 4. Initialise the database (first time only)

```bash
docker compose -f airflow/docker-compose.yaml up airflow-init
```

Wait until you see `airflow-init-1 exited with code 0` before proceeding.

---

## Starting Airflow

```bash
docker compose -f airflow/docker-compose.yaml up
```

Open the UI at **http://localhost:8081** and log in with `admin` / `admin`.

The scheduler automatically picks up any `.py` file in `airflow/dags/` that
contains a `DAG` object. Both DAGs will appear in the UI within 30 seconds.

---

## Stopping Airflow

```bash
docker compose -f airflow/docker-compose.yaml down
```

To also remove the database volume (full reset):

```bash
docker compose -f airflow/docker-compose.yaml down -v
```

---

## Running DAGs

### Manually trigger from the UI

1. Open http://localhost:8081
2. Find the DAG (`crop_disease_training_pipeline` or `crop_disease_monitoring`)
3. Click the toggle to unpause it
4. Click **▶ Trigger DAG** to run immediately

### Manually trigger from the CLI

```bash
# training pipeline
docker compose -f airflow/docker-compose.yaml exec airflow-scheduler \
    airflow dags trigger crop_disease_training_pipeline

# monitoring
docker compose -f airflow/docker-compose.yaml exec airflow-scheduler \
    airflow dags trigger crop_disease_monitoring
```

### Schedules

| DAG | Schedule | Runs |
|---|---|---|
| `crop_disease_training_pipeline` | `0 0 * * 0` | Every Sunday midnight |
| `crop_disease_monitoring` | `0 6 * * *` | Every day at 6am |

---

## DAG: training_pipeline

```
preprocess_data
      │
upload_data_to_s3
      │
  train_model          ← runs src/model/train.py, captures RUN_ID from stdout
      │
evaluate_threshold     ← checks test_f1 ≥ 0.85
    /       \
register   notify_low_performance
    │
update_ssm             ← writes RUN_ID to /crop-disease-mlops/staging/run_id
    │
deploy_lambda          ← updates Lambda env vars with new RUN_ID
    │
   end
```

The F1 threshold is set in `dag_tasks.py`:
```python
F1_THRESHOLD = 0.85
```

### XCom flow

`train_model` (BashOperator) captures `stdout` and pushes it to XCom.
`evaluate_threshold` reads it, parses the `RUN_ID=<value>` line, and
passes the run_id downstream via `ti.xcom_push(key="run_id", value=run_id)`.

---

## DAG: monitoring

```
compute_drift_metrics   ← runs src/monitoring/drift.py, writes to PostgreSQL
        │
check_drift_threshold   ← reads latest metrics from DB
      /       \
trigger_retraining   log_healthy
```

Drift thresholds (set in `monitoring_dag.py`):

| Metric | Threshold |
|---|---|
| `prediction_drift_score` | > 0.15 |
| `num_drifted_columns` | > 5 |
| `share_missing_values` | > 0.05 |

If any threshold is exceeded, `TriggerDagRunOperator` fires
`crop_disease_training_pipeline` automatically.

---

## Testing the DAG logic

Unit tests for the Python callables in `dag_tasks.py` run without Airflow:

```bash
pytest tests/unit/test_training_dag.py -k "not TestDagStructure" -v
```

DAG structure tests (requires Airflow running):

```bash
pytest tests/unit/test_training_dag.py -v
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Permission denied: /opt/airflow/logs` | `sudo rm -rf airflow/logs && mkdir -p airflow/logs` |
| `AIRFLOW_UID` not set | `echo "AIRFLOW_UID=$(id -u)" > airflow/.env` |
| DAG not appearing in UI | Check `airflow/dags/` for syntax errors: `python airflow/dags/training_pipeline.py` |
| `train_model` task fails | Check that `src/model/train.py` runs locally first |
| SSM update fails | Check EC2 IAM role has `ssm:PutParameter` permission |
| Lambda deploy fails | Check IAM role has `lambda:UpdateFunctionConfiguration` permission |

---

## Cost management

Airflow runs entirely locally (Docker) — no AWS charges for the orchestration
itself. Charges only occur when DAG tasks interact with AWS:

- **S3 sync** — minimal (~$0.005 per GB)
- **EC2 training** — ~$0.53/hour while `train_model` task runs
- **Lambda update** — free (configuration change only)
- **SSM** — free tier covers standard parameters

Stop EC2 and RDS after training to avoid idle charges.

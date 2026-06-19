# Airflow Orchestration — Setup Guide

This guide documents how to run the Airflow orchestration stack locally.
Airflow manages two DAGs:

- **`crop_disease_training_pipeline`** — weekly: preprocess → upload to S3 → train on EC2 → evaluate → deploy
- **`crop_disease_monitoring`** — daily: compute drift metrics → trigger retraining if needed

---

## Prerequisites

- Docker and Docker Compose installed
- AWS CLI configured (`aws configure`)
- MLflow server running on EC2 (see `notebooks/README.md`)
- PEM key for the training EC2 instance
- Training EC2 instance set up (see `airflow/setup_training_ec2.sh`)
- Processed data in S3

---

## Directory structure

```
airflow/
├── docker-compose.yaml       ← Airflow stack (scheduler + webserver + postgres)
├── .env                      ← auto-generated, contains AIRFLOW_UID + connection vars
├── keys/
│   └── training_key.pem   ← PEM key for training EC2 (gitignored)
├── dags/
│   ├── training_pipeline.py  ← weekly training DAG
│   ├── monitoring_dag.py     ← daily monitoring DAG
│   └── dag_tasks.py          ← Python callables (no Airflow dependency)
├── logs/                     ← task execution logs (auto-created)
└── plugins/                  ← custom Airflow plugins (empty for now)
```

---

## First-time setup

### 0. Set up the training EC2 instance

Copy the setup script to EC2 and run it:
```bash
scp -i your-key.pem infrastructure/scripts/setup_training_ec2.sh \
    ec2-user@YOUR_EC2_DNS:~/

ssh -i your-key.pem ec2-user@YOUR_EC2_DNS
bash setup_training_ec2.sh
```

### 1. Copy the training EC2 key

```bash
mkdir -p airflow/keys
cp /path/to/your-training-key.pem airflow/keys/training_key.pem
chmod 400 airflow/keys/training_key.pem
```

### 2. Create required directories

```bash
sudo rm -rf airflow/logs airflow/plugins   # remove if already created by Docker
mkdir -p airflow/logs airflow/plugins
```

### 3. Create the .env file

```bash
echo "AIRFLOW_UID=$(id -u)" > airflow/.env
echo "TRAINING_EC2_DNS=ec2-xx-xx-xx-xx.eu-west-1.compute.amazonaws.com" >> airflow/.env
echo "MLFLOW_TRACKING_URI=http://ec2-52-211-42-124.eu-west-1.compute.amazonaws.com:5000" >> airflow/.env
echo "MODEL_BUCKET=crop-disease-models-mlflow-stg-478544568263" >> airflow/.env
echo "AWS_DEFAULT_REGION=eu-west-1" >> airflow/.env
echo "LAMBDA_FUNCTION=crop-disease-predict_crop-disease-mlops" >> airflow/.env
```

Replace `TRAINING_EC2_DNS` with your actual training EC2 public DNS.

### 4. Initialise the database (first time only)

```bash
docker compose -f airflow/docker-compose.yaml up airflow-init
```

Wait until you see `airflow-init-1 exited with code 0`.

This step automatically:
- Installs `apache-airflow-providers-ssh` and other dependencies
- Creates the admin user (admin/admin)
- Registers the SSH connection for the training EC2 via the `AIRFLOW_CONN_GPU_EC2_TRAINING` env var — no manual UI setup needed

---

## Starting Airflow

```bash
docker compose -f airflow/docker-compose.yaml up
```

Open the UI at **http://localhost:8081** and log in with `admin` / `admin`.

### 5. Verify SSH connection to training EC2

```bash
docker exec -it airflow-airflow-scheduler-1 \
    ssh -i /opt/airflow/keys/training_key.pem \
    -o StrictHostKeyChecking=no \
    ec2-user@YOUR_TRAINING_EC2_DNS \
    "echo 'SSH connection works'"
```

---

## Stopping Airflow

```bash
docker compose -f airflow/docker-compose.yaml down
```

Full reset (removes database):
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
docker-compose -f airflow/docker-compose.yaml exec airflow-scheduler \
    airflow dags trigger crop_disease_training_pipeline

# monitoring
docker-compose -f airflow/docker-compose.yaml exec airflow-scheduler \
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
preprocess_data          ← runs locally in Airflow container
      │
upload_data_to_s3        ← syncs data/processed/ to S3
      │
  train_model            ← SSHOperator → training EC2
      │                    runs src/model/train.py, prints RUN_ID to stdout
      │
evaluate_threshold       ← checks test_f1 ≥ 0.85 in MLflow
    /       \
register   notify_low_performance
    │
update_ssm               ← writes RUN_ID to SSM Parameter Store
    │
deploy_lambda            ← updates Lambda env vars with new RUN_ID
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

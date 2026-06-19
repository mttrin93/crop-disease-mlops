"""
Crop Disease Detection — Training Pipeline DAG

Orchestrates the full training workflow on a weekly schedule:

  preprocess_data
        │
  upload_data_to_s3
        │
    train_model
        │
  evaluate_threshold  ──── [below threshold] ──→  notify_low_performance
        │
  [above threshold]
        │
  register_model
        │
   update_ssm
        │
  deploy_lambda

Schedule: weekly (Sunday midnight)
Manual trigger: available via Airflow UI
"""

import base64
import logging
from datetime import datetime, timedelta

import boto3
import mlflow
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.ssh.operators.ssh import SSHOperator

from airflow import DAG

logger = logging.getLogger(__name__)

# ── DAG default args ──────────────────────────────────────────────────────────

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# ── constants (set via Airflow Variables or env vars in docker-compose) ────────

import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "crop-disease-models-stg-478544568263")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
MODEL_NAME = "crop-disease-efficientnet-b0"
SSM_PARAMETER = "/crop-disease-mlops/staging/run_id"
LAMBDA_FUNCTION = os.getenv("LAMBDA_FUNCTION", "crop-disease-predict_mlops-zoomcamp")
F1_THRESHOLD = 0.85  # minimum test F1 to promote model to production

# SSH connection ID configured in Airflow UI:
# Admin → Connections → gpu_ec2_training
# Host: your GPU EC2 public DNS
# Login: ec2-user
# Private key: contents of your .pem file
SSH_CONN_ID = "gpu_ec2_training"

# remote paths on the GPU EC2 instance
REMOTE_REPO_DIR = "/home/ec2-user/crop-disease-mlops"
REMOTE_DATA_DIR = "/tmp/crop_disease_data/processed"

# ── training command (runs on GPU EC2 via SSH) ─────────────────────────────────

TRAIN_COMMAND = f"""
set -e

# pull latest code
cd {REMOTE_REPO_DIR}
git pull origin develop

# download processed data from S3
mkdir -p {REMOTE_DATA_DIR}
aws s3 sync s3://{MODEL_BUCKET}/data/processed {REMOTE_DATA_DIR} \
    --region {AWS_REGION}

# run training — prints RUN_ID=<value> to stdout on completion
python src/model/train.py \
    --data-dir {REMOTE_DATA_DIR} \
    --epochs 15 \
    --batch-size 64 \
    --mlflow-uri {MLFLOW_TRACKING_URI} \
    --experiment-name crop-disease-detection \
    --model-name {MODEL_NAME} \
    --model-bucket {MODEL_BUCKET}
"""

# ── task functions ─────────────────────────────────────────────────────────────


def _evaluate_threshold(**context) -> str:
    """
    Branch operator: check if the trained model meets the F1 threshold.
    Reads the run_id from XCom (pushed by train_model task).
    Returns the task_id of the next task to execute.
    """
    ti = context["ti"]
    train_output = ti.xcom_pull(task_ids="train_model")

    # SSHOperator base64-encodes stdout — decode it first
    try:
        train_output = base64.b64decode(train_output).decode("utf-8")
    except Exception:
        pass

    # parse RUN_ID from train.py stdout: "RUN_ID=abc123"
    run_id = None
    for line in train_output.splitlines():
        if line.startswith("RUN_ID="):
            run_id = line.split("=", 1)[1].strip()
            break

    if not run_id:
        raise ValueError(
            f"Could not parse RUN_ID from train_model output: {train_output}"
        )

    ti.xcom_push(key="run_id", value=run_id)
    logger.info(f"Evaluating run {run_id}...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    test_f1 = run.data.metrics.get("test_f1", 0.0)

    logger.info(f"Test F1: {test_f1:.4f} | Threshold: {F1_THRESHOLD}")

    if test_f1 >= F1_THRESHOLD:
        return "register_model"
    return "notify_low_performance"


def _register_model(**context) -> None:
    """Promote model to Staging then Production in MLflow Model Registry."""
    ti = context["ti"]
    run_id = ti.xcom_pull(task_ids="evaluate_threshold", key="run_id")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # get the latest version for this run
    versions = client.search_model_versions(f"run_id='{run_id}'")
    if not versions:
        raise ValueError(f"No model versions found for run_id={run_id}")

    version = versions[0].version
    logger.info(f"Promoting version {version} to Production...")

    client.set_registered_model_alias(name=MODEL_NAME, version=version, alias="Staging")
    client.set_registered_model_alias(
        name=MODEL_NAME,
        version=version,
        alias="Production",
        # archive_existing_versions=True,
    )
    logger.info(f"Model version {version} is now in Production")


def _update_ssm(**context) -> None:
    """Write the active RUN_ID to SSM Parameter Store."""
    ti = context["ti"]
    run_id = ti.xcom_pull(task_ids="evaluate_threshold", key="run_id")

    ssm = boto3.client("ssm", region_name=AWS_REGION)
    ssm.put_parameter(
        Name=SSM_PARAMETER,
        Value=run_id,
        Type="String",
        Overwrite=True,
    )
    logger.info(f"SSM {SSM_PARAMETER} updated → {run_id}")


def _deploy_lambda(**context) -> None:
    """Update Lambda environment variables with the new RUN_ID."""
    ti = context["ti"]
    run_id = ti.xcom_pull(task_ids="evaluate_threshold", key="run_id")

    lambda_client = boto3.client("lambda", region_name=AWS_REGION)

    # wait for Lambda to finish any in-progress update
    import time

    for _ in range(12):
        response = lambda_client.get_function(FunctionName=LAMBDA_FUNCTION)
        status = response["Configuration"]["LastUpdateStatus"]
        if status != "InProgress":
            break
        logger.info("Lambda update in progress, waiting...")
        time.sleep(5)

    lambda_client.update_function_configuration(
        FunctionName=LAMBDA_FUNCTION,
        Environment={
            "Variables": {
                "RUN_ID": run_id,
                "MODEL_BUCKET": MODEL_BUCKET,
                "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            }
        },
    )
    logger.info(f"Lambda {LAMBDA_FUNCTION} updated with RUN_ID={run_id}")


# ── DAG definition ─────────────────────────────────────────────────────────────

with DAG(
    dag_id="crop_disease_training_pipeline",
    default_args=default_args,
    description="Weekly training pipeline for crop disease detection model",
    schedule="0 0 * * 0",  # every Sunday at midnight
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "training", "crop-disease"],
) as dag:

    # ── 1. preprocess raw images ───────────────────────────────────────────────
    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command=(
            "cd /opt/airflow && " "python src/data/preprocess.py --skip-upload"
        ),
    )

    # ── 2. upload processed splits to S3 ──────────────────────────────────────
    upload_data_to_s3 = BashOperator(
        task_id="upload_data_to_s3",
        bash_command=(
            f"aws s3 sync data/processed/ "
            f"s3://{MODEL_BUCKET}/data/processed/ "
            f"--region {AWS_REGION}"
        ),
    )

    #    # ── 3. train model (captures stdout to XCom for RUN_ID) ───────────────────
    #    train_model = BashOperator(
    #        task_id="train_model",
    #        bash_command=(
    #            "cd /opt/airflow && "
    #            "python src/model/train.py "
    #            f"    --data-dir data/processed "
    #            f"    --epochs 15 "
    #            f"    --batch-size 64 "
    #            f"    --mlflow-uri {MLFLOW_TRACKING_URI} "
    #            f"    --experiment-name crop-disease-detection "
    #            f"    --model-name {MODEL_NAME}"
    #        ),
    #        #        bash_command="echo 'RUN_ID=afe0821a13c74ed2a10429da27ac4577'",
    #        do_xcom_push=True,  # captures stdout → XCom for downstream tasks
    #    )

    # ── 3. train model on GPU EC2 via SSH ─────────────────────────────────────
    #
    # SSHOperator connects to the GPU EC2 instance and runs the training
    # command remotely. Airflow captures stdout via XCom to extract RUN_ID.
    #
    # The EC2 instance must be:
    #   - running (start it before triggering the DAG)
    #   - configured with the repo cloned and training env installed
    #     (see infrastructure/scripts/setup_training_ec2.sh)
    #   - configured as an Airflow SSH connection (gpu_ec2_training)
    #
    train_model = SSHOperator(
        task_id="train_model",
        ssh_conn_id=SSH_CONN_ID,
        command=TRAIN_COMMAND,
        do_xcom_push=True,  # captures stdout → XCom (contains RUN_ID=...)
        cmd_timeout=7200,  # 2 hours max for training
        get_pty=True,  # allocate pseudo-terminal for better output
    )

    # ── 4. check if model meets F1 threshold ──────────────────────────────────
    evaluate_threshold = BranchPythonOperator(
        task_id="evaluate_threshold",
        python_callable=_evaluate_threshold,
    )

    # ── 5a. model below threshold → notify and stop ───────────────────────────
    notify_low_performance = BashOperator(
        task_id="notify_low_performance",
        bash_command=(
            "echo 'Model did not meet F1 threshold "
            f"({F1_THRESHOLD}). Skipping deployment.'"
        ),
    )

    # ── 5b. model above threshold → register → update SSM → deploy ───────────
    register_model = PythonOperator(
        task_id="register_model",
        python_callable=_register_model,
    )

    update_ssm = PythonOperator(
        task_id="update_ssm",
        python_callable=_update_ssm,
    )

    deploy_lambda = PythonOperator(
        task_id="deploy_lambda",
        python_callable=_deploy_lambda,
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success",
    )

    # ── dependencies ──────────────────────────────────────────────────────────
    (
        preprocess_data
        >> upload_data_to_s3
        >> train_model
        >> evaluate_threshold
        >> [register_model, notify_low_performance]
    )

    register_model >> update_ssm >> deploy_lambda >> end
    notify_low_performance >> end

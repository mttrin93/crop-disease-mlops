"""
Crop Disease Detection — Monitoring DAG

Runs daily to detect data and prediction drift using Evidently.
Triggers retraining if drift exceeds threshold.

  compute_drift_metrics
          │
  check_drift_threshold
       /        \
  [drift]    [no drift]
     │             │
trigger_retraining  log_healthy

Schedule: daily at 6am
"""

import os
import logging
from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from airflow import DAG

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "crop-disease-models-stg-478544568263")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")

# drift thresholds — retrain if any is exceeded
DRIFT_SCORE_THRESHOLD = 0.15  # prediction drift score
DRIFTED_COLUMNS_THRESHOLD = 5  # number of drifted input features
MISSING_VALUES_THRESHOLD = 0.05  # share of missing values


def _check_drift_threshold(**context) -> str:
    """
    Read drift metrics written to PostgreSQL by the monitoring script
    and decide whether to trigger retraining.
    Returns task_id to execute next.
    """
    import psycopg2

    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB", "monitoring"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "example"),
    )

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT prediction_drift, num_drifted_columns, share_missing_values
            FROM drift_metrics
            ORDER BY timestamp DESC
            LIMIT 1
        """
        )
        row = cur.fetchone()

    conn.close()

    if row is None:
        logger.warning("No drift metrics found in DB — skipping threshold check")
        return "log_healthy"

    drift_score, num_drifted, missing_values = row
    logger.info(
        f"Latest metrics: drift_score={drift_score:.4f}, "
        f"drifted_cols={num_drifted}, missing={missing_values:.4f}"
    )

    if (
        drift_score > DRIFT_SCORE_THRESHOLD
        or num_drifted > DRIFTED_COLUMNS_THRESHOLD
        or missing_values > MISSING_VALUES_THRESHOLD
    ):
        logger.warning(f"Drift threshold exceeded — triggering retraining")
        return "trigger_retraining"

    logger.info("All metrics within thresholds — no retraining needed")
    return "log_healthy"


default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="crop_disease_monitoring",
    default_args=default_args,
    description="Daily drift monitoring for crop disease detection model",
    schedule="0 6 * * *",  # every day at 6am
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "monitoring", "crop-disease"],
) as dag:

    # ── 1. compute Evidently drift metrics → write to PostgreSQL ──────────────
    compute_drift_metrics = BashOperator(
        task_id="compute_drift_metrics",
        bash_command=(
            "cd /opt/airflow && "
            "python src/monitoring/drift.py "
            f"    --model-bucket {MODEL_BUCKET} "
            f"    --region {AWS_REGION}"
        ),
    )

    # ── 2. check metrics against thresholds ───────────────────────────────────
    check_drift_threshold = BranchPythonOperator(
        task_id="check_drift_threshold",
        python_callable=_check_drift_threshold,
    )

    # ── 3a. drift detected → trigger retraining DAG ───────────────────────────
    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="crop_disease_training_pipeline",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    # ── 3b. no drift → log healthy ────────────────────────────────────────────
    log_healthy = BashOperator(
        task_id="log_healthy",
        bash_command="echo 'All drift metrics within thresholds. Model healthy.'",
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success",
    )

    # ── dependencies ──────────────────────────────────────────────────────────
    (
        compute_drift_metrics
        >> check_drift_threshold
        >> [trigger_retraining, log_healthy]
        >> end
    )

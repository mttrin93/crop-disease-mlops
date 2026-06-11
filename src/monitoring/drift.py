"""
Batch drift monitoring for the crop disease detection model.

Compares a recent window of API predictions against the reference distribution
computed at training time, using Evidently. Writes drift metrics to PostgreSQL
so Grafana can display them.

What we monitor (prediction-based drift — not raw images):
  - predicted class distribution     → did the model start seeing different crops?
  - confidence score distribution    → is the model less certain than at training?
  - prediction entropy               → higher entropy = model is more uncertain

Reference data: a sample of test-set predictions saved to S3 at training time
Current data:   predictions logged to PostgreSQL by the API (last window_days)

Called by the Airflow monitoring DAG (airflow/dags/monitoring_dag.py).

Usage (standalone):
    python src/monitoring/drift.py --window-days 7
    python src/monitoring/drift.py --window-days 7 --skip-db  # print only
"""

import io
import os
import logging
import argparse
from datetime import datetime, timezone, timedelta

import boto3
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
)
logger = logging.getLogger(__name__)

# ── configuration ─────────────────────────────────────────────────────────────

MODEL_BUCKET = os.getenv("MODEL_BUCKET", "crop-disease-models-stg-478544568263")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-west-1")
REFERENCE_S3_KEY = "monitoring/reference_predictions.parquet"

DB_HOST = os.getenv("MONITORING_DB_HOST", "localhost")
DB_PORT = int(os.getenv("MONITORING_DB_PORT", "5432"))
DB_NAME = os.getenv("MONITORING_DB_NAME", "monitoring")
DB_USER = os.getenv("MONITORING_DB_USER", "postgres")
DB_PASSWORD = os.getenv("MONITORING_DB_PASSWORD", "example")

# columns Evidently will analyse
NUMERICAL_FEATURES = ["confidence", "entropy"]
CATEGORICAL_FEATURES = ["predicted_class"]


# ── PostgreSQL helpers ────────────────────────────────────────────────────────


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def ensure_tables_exist(conn) -> None:
    """Create tables if they don't exist yet."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id               SERIAL PRIMARY KEY,
                timestamp        TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                predicted_class  VARCHAR(100),
                confidence       FLOAT,
                entropy          FLOAT,
                run_id           VARCHAR(64)
            );
            CREATE TABLE IF NOT EXISTS drift_metrics (
                id                    SERIAL PRIMARY KEY,
                timestamp             TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                prediction_drift      FLOAT,
                num_drifted_columns   INTEGER,
                share_missing_values  FLOAT,
                window_size           INTEGER
            );
        """
        )
    conn.commit()
    logger.info("Tables verified / created")


def load_current_predictions(conn, window_days: int) -> pd.DataFrame:
    """
    Load the last window_days of predictions from PostgreSQL.
    Falls back to a small synthetic sample if the table is empty (for demo).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT predicted_class, confidence, entropy, timestamp
            FROM predictions
            WHERE timestamp >= %s
            ORDER BY timestamp DESC
        """,
            (cutoff,),
        )
        rows = cur.fetchall()

    if not rows:
        logger.warning(
            "No predictions found in the last %d days. "
            "Using synthetic data for demonstration.",
            window_days,
        )
        return _synthetic_current_data()

    df = pd.DataFrame(
        rows, columns=["predicted_class", "confidence", "entropy", "timestamp"]
    )
    logger.info("Loaded %d current predictions (last %d days)", len(df), window_days)
    return df


def _synthetic_current_data(n: int = 200) -> pd.DataFrame:
    """
    Generate synthetic prediction data for demonstration when the
    predictions table is empty. Introduces slight drift vs reference.
    """
    rng = np.random.default_rng(seed=42)
    classes = [
        "Tomato___healthy",
        "Tomato___Late_blight",
        "Apple___Apple_scab",
        "Apple___healthy",
        "Corn_(maize)___Common_rust_",
        "Potato___Early_blight",
        "Grape___Black_rot",
    ]
    # bias toward a few classes to simulate distribution drift
    weights = [0.35, 0.25, 0.15, 0.10, 0.08, 0.04, 0.03]
    return pd.DataFrame(
        {
            "predicted_class": rng.choice(classes, size=n, p=weights),
            "confidence": rng.beta(a=5, b=2, size=n).clip(0.4, 1.0),
            "entropy": rng.beta(a=2, b=5, size=n),
        }
    )


def write_drift_metrics(
    conn,
    prediction_drift: float,
    num_drifted_columns: int,
    share_missing_values: float,
    window_size: int,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO drift_metrics
                (timestamp, prediction_drift, num_drifted_columns,
                 share_missing_values, window_size)
            VALUES (NOW(), %s, %s, %s, %s)
        """,
            (prediction_drift, num_drifted_columns, share_missing_values, window_size),
        )
    conn.commit()
    logger.info(
        "Drift metrics written: drift=%.4f, drifted_cols=%d, missing=%.4f",
        prediction_drift,
        num_drifted_columns,
        share_missing_values,
    )


# ── reference data ────────────────────────────────────────────────────────────


def load_reference_data(bucket: str, key: str) -> pd.DataFrame:
    """
    Download reference prediction statistics from S3.
    Generated at training time and uploaded once.
    """
    s3 = boto3.client("s3", region_name=AWS_REGION)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        logger.info(
            "Reference data loaded from s3://%s/%s (%d rows)", bucket, key, len(df)
        )
        return df
    except s3.exceptions.NoSuchKey:
        logger.warning("Reference data not found in S3 — using synthetic reference")
        return _synthetic_reference_data()


def _synthetic_reference_data(n: int = 500) -> pd.DataFrame:
    """Generate synthetic reference data (balanced class distribution)."""
    rng = np.random.default_rng(seed=0)
    classes = [
        "Tomato___healthy",
        "Tomato___Late_blight",
        "Apple___Apple_scab",
        "Apple___healthy",
        "Corn_(maize)___Common_rust_",
        "Potato___Early_blight",
        "Grape___Black_rot",
    ]
    return pd.DataFrame(
        {
            "predicted_class": rng.choice(classes, size=n),
            "confidence": rng.beta(a=8, b=2, size=n).clip(0.5, 1.0),
            "entropy": rng.beta(a=2, b=8, size=n),
        }
    )


# ── Evidently report ─────────────────────────────────────────────────────────


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[float, int, float]:
    """
    Run Evidently DataDrift report.
    Returns (prediction_drift_score, num_drifted_columns, share_missing_values).

    Compatible with evidently>=0.4 (uses legacy API for v0.7+).
    Result structure: metrics[0] = DatasetDriftMetric, metrics[1] = DataDriftTable.
    """
    column_mapping = ColumnMapping(
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference[NUMERICAL_FEATURES + CATEGORICAL_FEATURES],
        current_data=current[NUMERICAL_FEATURES + CATEGORICAL_FEATURES],
        column_mapping=column_mapping,
    )

    result = report.as_dict()

    # DatasetDriftMetric result keys:
    # drift_share, number_of_columns, number_of_drifted_columns,
    # share_of_drifted_columns, dataset_drift
    dataset_result = result["metrics"][0]["result"]
    prediction_drift = dataset_result["share_of_drifted_columns"]
    num_drifted_columns = dataset_result["number_of_drifted_columns"]

    # compute missing values directly from pandas (simpler and version-independent)
    features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    share_missing_values = float(current[features].isnull().mean().mean())

    logger.info(
        "Evidently report: drift_share=%.4f, drifted_cols=%d, missing=%.4f",
        prediction_drift,
        num_drifted_columns,
        share_missing_values,
    )
    return prediction_drift, num_drifted_columns, share_missing_values


# ── main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute prediction drift metrics")
    parser.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="Days of recent predictions to analyse (default: 7)",
    )
    parser.add_argument("--model-bucket", type=str, default=MODEL_BUCKET)
    parser.add_argument("--region", type=str, default=AWS_REGION)
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Print metrics only, do not write to PostgreSQL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # load reference data
    reference_df = load_reference_data(args.model_bucket, REFERENCE_S3_KEY)

    # load current predictions and compute drift
    if args.skip_db:
        current_df = _synthetic_current_data()
        prediction_drift, num_drifted, missing = compute_drift_report(
            reference_df, current_df
        )
        print(f"prediction_drift={prediction_drift:.4f}")
        print(f"num_drifted_columns={num_drifted}")
        print(f"share_missing_values={missing:.4f}")
        return

    conn = get_db_connection()
    ensure_tables_exist(conn)

    current_df = load_current_predictions(conn, window_days=args.window_days)
    prediction_drift, num_drifted, missing = compute_drift_report(
        reference_df, current_df
    )

    write_drift_metrics(
        conn,
        prediction_drift=prediction_drift,
        num_drifted_columns=num_drifted,
        share_missing_values=missing,
        window_size=len(current_df),
    )
    conn.close()


if __name__ == "__main__":
    main()

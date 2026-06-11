"""Unit tests for src/monitoring/drift.py."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import (
    write_drift_metrics,
    compute_drift_report,
    _synthetic_current_data,
    _synthetic_reference_data,
)

# ---------- synthetic data fixtures ----------


@pytest.fixture()
def reference_df() -> pd.DataFrame:
    return _synthetic_reference_data(n=200)


@pytest.fixture()
def current_df() -> pd.DataFrame:
    return _synthetic_current_data(n=100)


# ---------- synthetic data ----------


def test_synthetic_reference_has_correct_columns():
    df = _synthetic_reference_data()
    assert set(df.columns) >= {"predicted_class", "confidence", "entropy"}


def test_synthetic_reference_confidence_range():
    df = _synthetic_reference_data()
    assert df["confidence"].between(0.0, 1.0).all()


def test_synthetic_reference_entropy_non_negative():
    df = _synthetic_reference_data()
    assert (df["entropy"] >= 0).all()


def test_synthetic_current_has_correct_columns():
    df = _synthetic_current_data()
    assert set(df.columns) >= {"predicted_class", "confidence", "entropy"}


def test_synthetic_data_reproducible():
    df1 = _synthetic_current_data(n=50)
    df2 = _synthetic_current_data(n=50)
    pd.testing.assert_frame_equal(df1, df2)


# ---------- compute_drift_report ----------


def test_compute_drift_report_returns_three_values(reference_df, current_df):
    drift_score, num_drifted, missing = compute_drift_report(reference_df, current_df)
    assert isinstance(drift_score, float)
    assert isinstance(num_drifted, int)
    assert isinstance(missing, float)


def test_compute_drift_report_drift_score_range(reference_df, current_df):
    drift_score, _, _ = compute_drift_report(reference_df, current_df)
    assert 0.0 <= drift_score <= 1.0


def test_compute_drift_report_num_drifted_non_negative(reference_df, current_df):
    _, num_drifted, _ = compute_drift_report(reference_df, current_df)
    assert num_drifted >= 0


def test_compute_drift_report_missing_values_range(reference_df, current_df):
    _, _, missing = compute_drift_report(reference_df, current_df)
    assert 0.0 <= missing <= 1.0


def test_compute_drift_identical_data_has_low_drift(reference_df):
    """Same data as reference and current should produce zero drifted columns."""
    drift_score, num_drifted, _ = compute_drift_report(
        reference_df, reference_df.copy()
    )
    assert drift_score == 0.0
    assert num_drifted == 0


def test_compute_drift_different_distribution_has_higher_drift(reference_df):
    """Heavily biased current data should produce higher drift score."""
    n = len(reference_df)
    biased_current = pd.DataFrame(
        {
            "predicted_class": ["Tomato___healthy"] * n,  # all same class
            "confidence": np.ones(n) * 0.99,
            "entropy": np.zeros(n),
        }
    )
    drift_score, _, _ = compute_drift_report(reference_df, biased_current)
    assert drift_score > 0.1


# ---------- write_drift_metrics ----------


def test_write_drift_metrics_inserts_row():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    write_drift_metrics(
        conn=mock_conn,
        prediction_drift=0.12,
        num_drifted_columns=2,
        share_missing_values=0.01,
        window_size=150,
    )

    mock_cursor.execute.assert_called_once()
    mock_conn.commit.assert_called_once()


def test_write_drift_metrics_passes_correct_values():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    write_drift_metrics(
        conn=mock_conn,
        prediction_drift=0.08,
        num_drifted_columns=1,
        share_missing_values=0.02,
        window_size=200,
    )

    args = mock_cursor.execute.call_args[0]
    assert args[1] == (0.08, 1, 0.02, 200)

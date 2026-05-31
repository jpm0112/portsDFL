"""Tests for regression metric utilities."""

import math

import numpy as np
import pytest

from ports_dfl.metrics.regression import all_metrics, mae, mape, r2, rmse, summarize_folds


def test_perfect_predictions_yield_zero_error() -> None:
    # When predictions equal the truth exactly, every metric hits its best value.
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert r2(y, y) == 1.0
    assert mape(y, y) == 0.0


def test_constant_prediction_against_known_targets() -> None:
    # Hand-computed example: predict a constant 25.0 so the exact errors are known.
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.full_like(y_true, 25.0)
    # Absolute errors |25-y| = 15,5,5,15 -> mean 10.
    assert mae(y_true, y_pred) == pytest.approx(10.0)
    # Recompute RMSE the same way the metric does, so the test documents the
    # formula and stays correct if the inputs change.
    expected_rmse = math.sqrt((15**2 + 5**2 + 5**2 + 15**2) / 4)
    assert rmse(y_true, y_pred) == pytest.approx(expected_rmse)


def test_all_metrics_returns_complete_dict() -> None:
    # Checks the shape of the result (keys + value types), not exact numbers.
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 2.0, 2.9])
    metrics = all_metrics(y, p)
    # Set comparison: a missing OR extra key fails the test.
    assert set(metrics.keys()) == {"mae", "rmse", "r2", "mape"}
    # Values must be plain Python floats so callers can rely on them for JSON.
    for v in metrics.values():
        assert isinstance(v, float)


def test_mape_handles_near_zero_targets() -> None:
    """MAPE should not blow up when y_true contains zeros (eps floor)."""
    # MAPE divides by |y_true|; a 0.0 target would divide-by-zero without the
    # eps floor in mape().
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.5, 1.0])
    val = mape(y_true, y_pred)
    # Finite result confirms the eps floor prevented the blow-up.
    assert math.isfinite(val)


def test_summarize_folds_reports_mean_and_std() -> None:
    # summarize_folds takes one metrics dict per CV fold and returns a DataFrame
    # with extra "mean" and "std" summary rows.
    fold_metrics = [
        {"mae": 10.0, "rmse": 14.0},
        {"mae": 11.0, "rmse": 15.0},
        {"mae": 12.0, "rmse": 16.0},
    ]
    df = summarize_folds(fold_metrics)
    assert "mean" in df.index
    assert "std" in df.index
    assert df.loc["mean", "mae"] == pytest.approx(11.0)
    # Sample std (ddof=1) of [10, 11, 12] is 1.0, matching summarize_folds.
    assert df.loc["std", "mae"] == pytest.approx(1.0)

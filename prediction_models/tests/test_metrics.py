"""Tests for regression metric utilities."""

import math

import numpy as np
import pytest

from ports_dfl.metrics.regression import all_metrics, mae, mape, r2, rmse, summarize_folds


def test_perfect_predictions_yield_zero_error() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert r2(y, y) == 1.0
    assert mape(y, y) == 0.0


def test_constant_prediction_against_known_targets() -> None:
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.full_like(y_true, 25.0)
    assert mae(y_true, y_pred) == pytest.approx(10.0)
    # RMSE = sqrt(mean([15^2, 5^2, 5^2, 15^2]))
    expected_rmse = math.sqrt((15**2 + 5**2 + 5**2 + 15**2) / 4)
    assert rmse(y_true, y_pred) == pytest.approx(expected_rmse)


def test_all_metrics_returns_complete_dict() -> None:
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 2.0, 2.9])
    metrics = all_metrics(y, p)
    assert set(metrics.keys()) == {"mae", "rmse", "r2", "mape"}
    for v in metrics.values():
        assert isinstance(v, float)


def test_mape_handles_near_zero_targets() -> None:
    """MAPE should not blow up when y_true contains zeros (eps floor)."""
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.5, 1.0])
    val = mape(y_true, y_pred)
    assert math.isfinite(val)


def test_summarize_folds_reports_mean_and_std() -> None:
    fold_metrics = [
        {"mae": 10.0, "rmse": 14.0},
        {"mae": 11.0, "rmse": 15.0},
        {"mae": 12.0, "rmse": 16.0},
    ]
    df = summarize_folds(fold_metrics)
    assert "mean" in df.index
    assert "std" in df.index
    assert df.loc["mean", "mae"] == pytest.approx(11.0)
    # Sample std of [10, 11, 12] is 1.0
    assert df.loc["std", "mae"] == pytest.approx(1.0)

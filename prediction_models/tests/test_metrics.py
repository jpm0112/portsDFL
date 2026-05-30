"""Tests for regression metric utilities."""

import math

import numpy as np
import pytest

# Import the functions under test from the source package. Each name here is a
# function defined in ports_dfl/metrics/regression.py.
from ports_dfl.metrics.regression import all_metrics, mae, mape, r2, rmse, summarize_folds


# pytest automatically discovers and runs any function whose name starts with
# "test_". `-> None` is a type hint saying this function returns nothing (test
# functions don't return a value; they pass by not raising and fail on a bad
# assert). No `self`/class needed: plain functions are enough.
def test_perfect_predictions_yield_zero_error() -> None:
    # When predictions equal the truth exactly, every error metric should hit
    # its "perfect" value. Here we pass the same array `y` as both arguments.
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # `assert <expr>` is the core of a pytest test: if <expr> is False the test
    # FAILS (and pytest shows the values); if True, execution simply continues.
    assert mae(y, y) == 0.0   # no absolute error
    assert rmse(y, y) == 0.0  # no root-mean-squared error
    assert r2(y, y) == 1.0    # R2 == 1.0 means a perfect fit
    assert mape(y, y) == 0.0  # no percentage error


def test_constant_prediction_against_known_targets() -> None:
    # Hand-computed example: predict the same constant (25.0) for every target,
    # so we know the exact errors and can check the formulas precisely.
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    # np.full_like makes an array the same shape/dtype as y_true, filled with 25.
    y_pred = np.full_like(y_true, 25.0)
    # Absolute errors are |25-10|,|25-20|,|25-30|,|25-40| = 15,5,5,15 -> mean 10.
    # pytest.approx allows tiny floating-point rounding differences instead of
    # demanding an exact bit-for-bit ==, which is unsafe for floats.
    assert mae(y_true, y_pred) == pytest.approx(10.0)
    # RMSE = sqrt(mean([15^2, 5^2, 5^2, 15^2]))
    # Recompute the expected value the same way the metric does, so the test
    # documents the formula and stays correct if the inputs ever change.
    expected_rmse = math.sqrt((15**2 + 5**2 + 5**2 + 15**2) / 4)
    assert rmse(y_true, y_pred) == pytest.approx(expected_rmse)


def test_all_metrics_returns_complete_dict() -> None:
    # all_metrics() bundles every metric into one dict; this test checks the
    # SHAPE of that result (keys present + value types), not exact numbers.
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 2.0, 2.9])
    metrics = all_metrics(y, p)
    # Compare the set of keys (order-independent) to the exact expected names.
    # Using a set means a missing OR extra key fails the test.
    assert set(metrics.keys()) == {"mae", "rmse", "r2", "mape"}
    # Loop over every value and assert it is a plain Python float (not, e.g., a
    # numpy float or string) so callers can rely on the type for JSON/printing.
    for v in metrics.values():
        assert isinstance(v, float)


def test_mape_handles_near_zero_targets() -> None:
    """MAPE should not blow up when y_true contains zeros (eps floor)."""
    # MAPE divides by |y_true|; a 0.0 target would normally divide-by-zero and
    # produce inf/NaN. The eps floor in mape() should keep the result finite.
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.5, 1.0])
    val = mape(y_true, y_pred)
    # math.isfinite is False for inf/NaN; asserting True confirms the eps floor
    # actually prevented the blow-up. (We don't check the exact magnitude here.)
    assert math.isfinite(val)


def test_summarize_folds_reports_mean_and_std() -> None:
    # summarize_folds() takes one metrics dict per cross-validation fold and
    # returns a DataFrame with extra "mean" and "std" summary rows.
    fold_metrics = [
        {"mae": 10.0, "rmse": 14.0},
        {"mae": 11.0, "rmse": 15.0},
        {"mae": 12.0, "rmse": 16.0},
    ]
    df = summarize_folds(fold_metrics)
    # df.index holds the row labels; check the two summary rows were added.
    assert "mean" in df.index
    assert "std" in df.index
    # df.loc["mean", "mae"] reads the value at row "mean", column "mae".
    # Mean of [10, 11, 12] is 11.0.
    assert df.loc["mean", "mae"] == pytest.approx(11.0)
    # Sample std of [10, 11, 12] is 1.0
    # (sample std uses N-1 in the denominator: sqrt(((-1)^2+0+1^2)/(3-1)) = 1.0,
    # matching ddof=1 used inside summarize_folds.)
    assert df.loc["std", "mae"] == pytest.approx(1.0)

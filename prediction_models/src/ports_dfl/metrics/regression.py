"""Regression metrics for service-time prediction.

All functions accept numpy arrays and return float scalars (or DataFrames in
the case of fold summaries). Matching scikit-learn's argument order
(``y_true``, ``y_pred``) for consistency.
"""

import numpy as np  # numerical arrays + math (np is the conventional nickname)
import pandas as pd  # tables (DataFrames); pd is the conventional nickname
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# `y_true: np.ndarray` is a type hint: it says "y_true is expected to be a numpy
# array". `-> float` says this function returns a plain Python float. Hints are
# documentation for humans/tools; Python does NOT enforce them at runtime.
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in target units (hours)."""
    # sklearn returns a numpy float; float(...) converts it to a plain Python
    # float so callers get a simple scalar (nicer for printing/JSON).
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-squared error in target units (hours)."""
    # MSE is the average squared error; sqrt brings it back to the original
    # units (hours), which is easier to interpret than squared hours.
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    # R2 = 1 means perfect fit; 0 means no better than predicting the mean;
    # it can go negative when the model is worse than the mean baseline.
    return float(r2_score(y_true, y_pred))


# `eps: float = 1e-6` gives the parameter a default value, so callers may omit
# it. 1e-6 is scientific notation for 0.000001.
def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Mean absolute percentage error.

    Args:
        eps: Floor for ``|y_true|`` to avoid division by zero.
    """
    # np.asarray converts the inputs to numpy arrays (no copy if already one)
    # and forces float dtype so the division below is true division, not
    # integer division.
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Denominator = |y_true|, but never smaller than eps. np.maximum compares
    # element-by-element, so any near-zero true value is replaced by eps to
    # avoid dividing by zero (which would give inf/NaN).
    denom = np.maximum(np.abs(y_true), eps)
    # All arithmetic here is vectorized (operates on the whole array at once):
    # per-element absolute percentage error, then averaged with np.mean.
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


# `-> dict[str, float]` means this returns a dictionary whose keys are strings
# and whose values are floats.
def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute all regression metrics in a single dict."""
    # A dict literal: each "name": value pair maps a metric name to its result.
    # This just reuses the functions above so the names stay in sync.
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


# `list[dict[str, float]]` = a list, where each element is a {str: float} dict
# (one dict of metrics per cross-validation fold).
def summarize_folds(fold_metrics: list[dict[str, float]]) -> pd.DataFrame:
    """Collapse per-fold metric dicts into a DataFrame with mean and std rows.

    Args:
        fold_metrics: List with one ``{metric_name: value}`` dict per fold.

    Returns:
        DataFrame indexed by ``fold_0 ... fold_{K-1}`` plus ``mean`` and
        ``std`` summary rows.
    """
    # Building a DataFrame from a list of dicts: each dict becomes one row, and
    # the dict keys become the columns (the metric names).
    df = pd.DataFrame(fold_metrics)
    # Rename the row labels to fold_0, fold_1, ... using a list comprehension.
    # `f"fold_{i}"` is an f-string: the {i} is replaced by the value of i.
    # range(len(df)) yields 0..number_of_rows-1.
    df.index = [f"fold_{i}" for i in range(len(df))]
    summary = pd.DataFrame(
        {
            # axis=0 means "reduce down each column", giving one mean/std per
            # metric across all folds.
            "mean": df.mean(axis=0),
            # ddof=1 uses the sample standard deviation (divide by N-1). Note:
            # with a single fold this yields NaN (N-1 = 0). See REPORTED.
            "std": df.std(axis=0, ddof=1),
        }
    # .T transposes: turns the two columns (mean, std) into two rows so they can
    # be stacked underneath the per-fold rows.
    ).T
    # Stack the per-fold rows and the mean/std summary rows into one table.
    # axis=0 = concatenate vertically (add rows).
    return pd.concat([df, summary], axis=0)

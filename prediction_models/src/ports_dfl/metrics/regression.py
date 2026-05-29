"""Regression metrics for service-time prediction.

All functions accept numpy arrays and return float scalars (or DataFrames in
the case of fold summaries). Matching scikit-learn's argument order
(``y_true``, ``y_pred``) for consistency.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in target units (hours)."""
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-squared error in target units (hours)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    return float(r2_score(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Mean absolute percentage error.

    Args:
        eps: Floor for ``|y_true|`` to avoid division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute all regression metrics in a single dict."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


def summarize_folds(fold_metrics: list[dict[str, float]]) -> pd.DataFrame:
    """Collapse per-fold metric dicts into a DataFrame with mean and std rows.

    Args:
        fold_metrics: List with one ``{metric_name: value}`` dict per fold.

    Returns:
        DataFrame indexed by ``fold_0 ... fold_{K-1}`` plus ``mean`` and
        ``std`` summary rows.
    """
    df = pd.DataFrame(fold_metrics)
    df.index = [f"fold_{i}" for i in range(len(df))]
    summary = pd.DataFrame(
        {
            "mean": df.mean(axis=0),
            "std": df.std(axis=0, ddof=1),
        }
    ).T
    return pd.concat([df, summary], axis=0)

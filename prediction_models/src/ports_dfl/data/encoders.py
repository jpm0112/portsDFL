"""Build the feature preprocessor (categorical encoding + numeric scaling).

:func:`build_preprocessor` returns an *unfitted* sklearn ``ColumnTransformer`` keyed
to the project's feature roles (``config``). Fit it on TRAIN folds only and
``transform`` val/test to avoid target leakage; the fitted object is joblib-picklable,
so the exact same transform is reused at inference time.
"""

from __future__ import annotations

from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from ports_dfl.config import (
    HIGH_CARDINALITY_CATEGORICAL,
    LOW_CARDINALITY_CATEGORICAL,
    NUMERIC_FEATURES,
)

# Numeric scaler name -> class. StandardScaler centres/scales by mean/std; RobustScaler
# uses median/IQR (less sensitive to the heavy right tail of service times).
_SCALERS = {"standard": StandardScaler, "robust": RobustScaler}


def build_preprocessor(
    categorical_strategy: str = "target",
    numeric_scaler: str = "standard",
) -> ColumnTransformer:
    """Construct an unfitted preprocessor for the vessel features.

    Low-cardinality categoricals are always one-hot encoded (few, cheap columns).
    High-cardinality categoricals are target-encoded by default (one column each,
    smoothed so unseen levels fall back to the prior) or one-hot when asked.

    Args:
        categorical_strategy: ``"target"`` (target-mean encoding for high-cardinality
            columns) or ``"onehot"`` (one-hot every categorical — much wider output).
        numeric_scaler: ``"standard"`` (StandardScaler) or ``"robust"`` (RobustScaler).

    Returns:
        An unfitted ``ColumnTransformer`` that emits a dense float array.

    Raises:
        ValueError: if ``categorical_strategy`` or ``numeric_scaler`` is unknown.
    """
    if categorical_strategy not in ("target", "onehot"):
        raise ValueError(
            f"categorical_strategy must be 'target' or 'onehot', got {categorical_strategy!r}"
        )
    if numeric_scaler not in _SCALERS:
        raise ValueError(
            f"numeric_scaler must be one of {sorted(_SCALERS)}, got {numeric_scaler!r}"
        )

    high_card = (
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if categorical_strategy == "onehot"
        else TargetEncoder()
    )
    return ColumnTransformer(
        [
            (
                "low_card",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                LOW_CARDINALITY_CATEGORICAL,
            ),
            ("high_card", high_card, HIGH_CARDINALITY_CATEGORICAL),
            ("numeric", _SCALERS[numeric_scaler](), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

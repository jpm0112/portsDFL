"""Load the training dataset and split it into features / target.

The CSV at ``config.DATA_PATH`` is already cleaned and feature-engineered by
``data_pipeline/``, so loading is a plain read plus a column-presence check.
Keeping it here lets the schema constants in ``config.py`` stay the single source
of truth for which columns are features vs. target.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ports_dfl.config import (
    ALL_FEATURES,
    DATA_PATH,
    HIGH_CARDINALITY_CATEGORICAL,
    LOW_CARDINALITY_CATEGORICAL,
    NUMERIC_FEATURES,
    TARGET_COL,
)


def load_training_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Read the training dataset CSV and verify the modelling columns are present.

    Args:
        path: CSV to read; defaults to ``config.DATA_PATH`` (itself overridable via
            the ``$PORTSDFL_DATA`` env var). An explicit path is honoured verbatim.

    Returns:
        The loaded DataFrame with all of its columns.

    Raises:
        FileNotFoundError: if the CSV does not exist.
        ValueError: if any required feature or target column is missing.
    """
    csv_path = Path(path) if path is not None else DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {csv_path}. "
            "Set $PORTSDFL_DATA or pass an explicit path."
        )
    df = pd.read_csv(csv_path)
    missing = [c for c in [*ALL_FEATURES, TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset {csv_path} is missing required columns: {missing}")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a loaded DataFrame into the feature matrix X and target Series y.

    Args:
        df: a DataFrame from :func:`load_training_dataset`.

    Returns:
        ``(X, y)`` where X has exactly ``config.ALL_FEATURES`` columns (the target
        is excluded, so it can never leak) and y is the ``config.TARGET_COL`` Series.
        Both are copies, so callers cannot mutate ``df`` through them.
    """
    return df[ALL_FEATURES].copy(), df[TARGET_COL].copy()


def feature_role_summary() -> dict[str, list[str]]:
    """Return the feature-role lists (low/high-cardinality categorical, numeric).

    Mirrors the ``config`` constants so the preprocessor and reports share one
    authoritative grouping. Returns fresh lists so callers can't mutate config.
    """
    return {
        "low_cardinality": list(LOW_CARDINALITY_CATEGORICAL),
        "high_cardinality": list(HIGH_CARDINALITY_CATEGORICAL),
        "numeric": list(NUMERIC_FEATURES),
    }

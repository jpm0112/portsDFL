"""Sanity checks for the dataset loader and feature schema."""

import numpy as np

from ports_dfl.config import (
    ALL_FEATURES,
    HIGH_CARDINALITY_CATEGORICAL,
    LOW_CARDINALITY_CATEGORICAL,
    NUMERIC_FEATURES,
    TARGET_COL,
)
from ports_dfl.data.loader import (
    feature_role_summary,
    load_training_dataset,
    split_features_target,
)


def test_load_returns_expected_shape(df) -> None:
    """Dataset has 5,589 rows × at least 18 columns (target + 17 features)."""
    assert df.shape[0] == 5589
    assert TARGET_COL in df.columns


def test_no_missing_in_modelling_columns(df) -> None:
    """Modelling columns must be free of NAs (preprocessing should have handled them)."""
    cols = ALL_FEATURES + [TARGET_COL]
    assert df[cols].isna().sum().sum() == 0


def test_target_is_positive_and_varied(df) -> None:
    """Service time is strictly positive and has substantial variability."""
    target = df[TARGET_COL]
    assert (target > 0).all()
    assert target.nunique() > 100
    assert 1.0 < target.mean() < 200.0


def test_feature_roles_cover_all_features() -> None:
    """The three role lists partition ALL_FEATURES with no overlap or omissions."""
    roles = feature_role_summary()
    union = set(roles["low_cardinality"]) | set(roles["high_cardinality"]) | set(roles["numeric"])
    assert union == set(ALL_FEATURES)
    # No double-counting
    assert (
        len(roles["low_cardinality"])
        + len(roles["high_cardinality"])
        + len(roles["numeric"])
        == len(ALL_FEATURES)
    )


def test_split_features_target_returns_aligned(Xy) -> None:
    X, y = Xy
    assert len(X) == len(y)
    assert TARGET_COL not in X.columns
    assert set(X.columns) == set(ALL_FEATURES)


def test_load_with_explicit_path(tmp_path) -> None:
    """Loader respects an explicit path argument."""
    df = load_training_dataset()
    out = tmp_path / "tiny.csv"
    df.head(100).to_csv(out, index=False)
    df_small = load_training_dataset(out)
    assert len(df_small) == 100


def test_role_lists_match_config_constants() -> None:
    """Sanity: feature_role_summary mirrors the config lists exactly."""
    roles = feature_role_summary()
    assert roles["low_cardinality"] == LOW_CARDINALITY_CATEGORICAL
    assert roles["high_cardinality"] == HIGH_CARDINALITY_CATEGORICAL
    assert roles["numeric"] == NUMERIC_FEATURES


def test_numeric_features_are_finite(df) -> None:
    """All numeric feature columns must be finite floats/ints."""
    for col in NUMERIC_FEATURES:
        assert np.isfinite(df[col]).all(), f"Non-finite values found in {col}"

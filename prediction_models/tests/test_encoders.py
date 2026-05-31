"""Tests for the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from ports_dfl.config import HIGH_CARDINALITY_CATEGORICAL, NUMERIC_FEATURES
from ports_dfl.data.encoders import build_preprocessor


def test_target_encoding_pipeline_fits_and_transforms(Xy) -> None:
    """Target-encoded preprocessor fits and transforms without errors."""
    X, y = Xy
    pre = build_preprocessor(categorical_strategy="target")
    out = pre.fit_transform(X, y)
    # Row count preserved (one output row per input row).
    assert out.shape[0] == len(X)
    assert np.all(np.isfinite(out))


def test_onehot_pipeline_produces_wider_output(Xy) -> None:
    """One-hot strategy yields more columns than target-encoding."""
    X, y = Xy
    pre_target = build_preprocessor(categorical_strategy="target").fit(X, y)
    pre_onehot = build_preprocessor(categorical_strategy="onehot").fit(X, y)
    out_target = pre_target.transform(X)
    out_onehot = pre_onehot.transform(X)
    # One-hot expands each category into many 0/1 columns, so it must be wider.
    assert out_onehot.shape[1] > out_target.shape[1]


def test_unseen_high_cardinality_levels_dont_break_transform(Xy) -> None:
    """Smoothed target encoder must handle unseen categorical levels in val fold."""
    X, y = Xy
    pre = build_preprocessor(categorical_strategy="target").fit(X, y)
    X_unseen = X.iloc[:10].copy()
    # Inject a level the encoder never saw during fit, simulating an unseen level.
    X_unseen.loc[:, HIGH_CARDINALITY_CATEGORICAL[0]] = "__UNSEEN_LEVEL_xyz__"
    out = pre.transform(X_unseen)
    # Encoder should fall back to a prior, producing finite numbers, not NaN/inf.
    assert np.all(np.isfinite(out))


def test_invalid_strategy_raises() -> None:
    with pytest.raises(ValueError):
        build_preprocessor(categorical_strategy="bogus")  # type: ignore[arg-type]


def test_invalid_scaler_raises() -> None:
    with pytest.raises(ValueError):
        build_preprocessor(numeric_scaler="bogus")  # type: ignore[arg-type]


def test_robust_scaler_changes_numeric_block(Xy) -> None:
    """Switching numeric_scaler changes the transformed numerics' scale."""
    X, y = Xy
    # Same categorical handling for both, so any difference is due to the scaler.
    standard = build_preprocessor(numeric_scaler="standard").fit_transform(X, y)
    robust = build_preprocessor(numeric_scaler="robust").fit_transform(X, y)
    assert not np.allclose(standard, robust)


def test_train_fit_val_transform_layout_consistent() -> None:
    """Fit on train, transform val -> identical feature LAYOUT (so a model can use
    both). NOTE: this only checks the train-only-fit / val-transform pattern yields
    a consistent column layout; it does NOT verify the encoder ignored val's target
    (the original name/docstring overclaimed). A real value-level leakage assertion
    is a TODO once a leaked-encoder oracle is wired in."""
    # Seeded for reproducible synthetic data.
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Sitio": rng.choice(["A", "B"], size=200),
            "Tipo nave (agrupado)": rng.choice(["X", "Y"], size=200),
            "covid_era": "during",
            "Puerto origen": rng.choice(["P1", "P2", "P3"], size=200),
            "Puerto destino": rng.choice(["P1", "P2", "P3"], size=200),
            "Servicio": rng.choice(["S1", "S2"], size=200),
            "Línea naviera": rng.choice(["L1", "L2"], size=200),
            "Agencia": rng.choice(["A1", "A2"], size=200),
        }
    )
    for col in NUMERIC_FEATURES:
        df[col] = rng.standard_normal(200)
    y = rng.standard_normal(200)
    pre = build_preprocessor()
    # Fit on train (first 150) alone, then transform val (last 50): fitting on
    # train only prevents target leakage from val into the encoders.
    out_train = pre.fit_transform(df.iloc[:150], y[:150])
    out_val = pre.transform(df.iloc[150:])
    # Train and val must share the same feature layout so a model can use both.
    assert out_train.shape[1] == out_val.shape[1]

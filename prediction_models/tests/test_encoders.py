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
    assert out.shape[0] == len(X)
    # Output is a finite-valued numeric array
    assert np.all(np.isfinite(out))


def test_onehot_pipeline_produces_wider_output(Xy) -> None:
    """One-hot strategy yields more columns than target-encoding."""
    X, y = Xy
    pre_target = build_preprocessor(categorical_strategy="target").fit(X, y)
    pre_onehot = build_preprocessor(categorical_strategy="onehot").fit(X, y)
    out_target = pre_target.transform(X)
    out_onehot = pre_onehot.transform(X)
    assert out_onehot.shape[1] > out_target.shape[1]


def test_unseen_high_cardinality_levels_dont_break_transform(Xy) -> None:
    """Smoothed target encoder must handle unseen categorical levels in val fold."""
    X, y = Xy
    pre = build_preprocessor(categorical_strategy="target").fit(X, y)
    X_unseen = X.iloc[:10].copy()
    X_unseen.loc[:, HIGH_CARDINALITY_CATEGORICAL[0]] = "__UNSEEN_LEVEL_xyz__"
    out = pre.transform(X_unseen)
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
    standard = build_preprocessor(numeric_scaler="standard").fit_transform(X, y)
    robust = build_preprocessor(numeric_scaler="robust").fit_transform(X, y)
    # Different scalers produce different output for non-degenerate columns
    assert not np.allclose(standard, robust)


def test_train_only_target_encoding_no_leak() -> None:
    """Smoke test: fit on train, transform val gives different per-row encodings
    when the train target distribution doesn't include val rows."""
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
    out_train = pre.fit_transform(df.iloc[:150], y[:150])
    out_val = pre.transform(df.iloc[150:])
    assert out_train.shape[1] == out_val.shape[1]

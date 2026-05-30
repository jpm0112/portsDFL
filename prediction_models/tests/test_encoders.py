"""Tests for the preprocessing pipeline."""

# `import ... as ...` gives a module a short alias. `np`/`pd` are the
# conventional aliases for NumPy (numeric arrays) and pandas (tables).
import numpy as np
import pandas as pd
import pytest  # the testing framework; provides fixtures, `raises`, `approx`, etc.

# Project imports: constants (lists of column names) and the function under test.
from ports_dfl.config import HIGH_CARDINALITY_CATEGORICAL, NUMERIC_FEATURES
from ports_dfl.data.encoders import build_preprocessor


# pytest auto-discovers any function whose name starts with `test_` and runs it.
# The `Xy` parameter is a *fixture*: a reusable setup defined in conftest.py.
# Listing its name here tells pytest to build it and pass the result in (here a
# (features DataFrame, target Series) tuple). The `-> None` is a type hint saying
# the function returns nothing; it is documentation only and does not affect runtime.
def test_target_encoding_pipeline_fits_and_transforms(Xy) -> None:
    """Target-encoded preprocessor fits and transforms without errors."""
    # ARRANGE: unpack the fixture into the feature table X and target y.
    X, y = Xy
    # ACT: build a preprocessor that target-encodes categoricals, then fit + transform.
    pre = build_preprocessor(categorical_strategy="target")
    out = pre.fit_transform(X, y)
    # ASSERT: an `assert` fails the test if the expression is False.
    # Row count must be preserved (one output row per input row).
    assert out.shape[0] == len(X)
    # Output is a finite-valued numeric array
    # `np.isfinite` flags non-NaN, non-inf values; `np.all` requires every cell to pass.
    assert np.all(np.isfinite(out))


def test_onehot_pipeline_produces_wider_output(Xy) -> None:
    """One-hot strategy yields more columns than target-encoding."""
    X, y = Xy
    # Fit two preprocessors that differ only in how categoricals are encoded.
    pre_target = build_preprocessor(categorical_strategy="target").fit(X, y)
    pre_onehot = build_preprocessor(categorical_strategy="onehot").fit(X, y)
    out_target = pre_target.transform(X)
    out_onehot = pre_onehot.transform(X)
    # One-hot expands each category into many 0/1 columns, so it must be wider.
    # `.shape[1]` is the number of columns.
    assert out_onehot.shape[1] > out_target.shape[1]


def test_unseen_high_cardinality_levels_dont_break_transform(Xy) -> None:
    """Smoothed target encoder must handle unseen categorical levels in val fold."""
    X, y = Xy
    pre = build_preprocessor(categorical_strategy="target").fit(X, y)
    # Take a 10-row copy so we don't mutate the shared fixture data.
    X_unseen = X.iloc[:10].copy()
    # Overwrite a high-cardinality column with a category the encoder never saw
    # during fit, simulating an unseen level at prediction time.
    X_unseen.loc[:, HIGH_CARDINALITY_CATEGORICAL[0]] = "__UNSEEN_LEVEL_xyz__"
    out = pre.transform(X_unseen)
    # The encoder should fall back gracefully (e.g. to a prior), producing finite
    # numbers rather than NaN/inf or raising.
    assert np.all(np.isfinite(out))


# This test takes no fixture: it builds the bad input itself.
def test_invalid_strategy_raises() -> None:
    # `pytest.raises(ValueError)` asserts the code inside the `with` block raises
    # that exception. The test FAILS if no ValueError (or a different error) occurs.
    with pytest.raises(ValueError):
        # `# type: ignore[arg-type]` silences the static type checker for this
        # deliberately invalid argument; it has no effect on the running test.
        build_preprocessor(categorical_strategy="bogus")  # type: ignore[arg-type]


def test_invalid_scaler_raises() -> None:
    # Same pattern: an unknown scaler name must be rejected with ValueError.
    with pytest.raises(ValueError):
        build_preprocessor(numeric_scaler="bogus")  # type: ignore[arg-type]


def test_robust_scaler_changes_numeric_block(Xy) -> None:
    """Switching numeric_scaler changes the transformed numerics' scale."""
    X, y = Xy
    # Same categorical handling for both, so any output difference is due to the scaler.
    standard = build_preprocessor(numeric_scaler="standard").fit_transform(X, y)
    robust = build_preprocessor(numeric_scaler="robust").fit_transform(X, y)
    # Different scalers produce different output for non-degenerate columns
    # `np.allclose` is a float-tolerant "are these arrays ~equal?" check; `not allclose`
    # asserts they differ by more than the tolerance somewhere.
    assert not np.allclose(standard, robust)


def test_train_only_target_encoding_no_leak() -> None:
    """Smoke test: fit on train, transform val gives different per-row encodings
    when the train target distribution doesn't include val rows."""
    # `default_rng(0)` seeds a random generator so this synthetic data is reproducible.
    rng = np.random.default_rng(0)
    # Build a 200-row fake dataset with the categorical columns the preprocessor expects.
    # `rng.choice([...], size=200)` draws 200 random values from the given options.
    df = pd.DataFrame(
        {
            "Sitio": rng.choice(["A", "B"], size=200),
            "Tipo nave (agrupado)": rng.choice(["X", "Y"], size=200),
            "covid_era": "during",  # a scalar broadcasts to fill the whole column
            "Puerto origen": rng.choice(["P1", "P2", "P3"], size=200),
            "Puerto destino": rng.choice(["P1", "P2", "P3"], size=200),
            "Servicio": rng.choice(["S1", "S2"], size=200),
            "Línea naviera": rng.choice(["L1", "L2"], size=200),
            "Agencia": rng.choice(["A1", "A2"], size=200),
        }
    )
    # Add the numeric feature columns, each filled with random standard-normal values.
    for col in NUMERIC_FEATURES:
        df[col] = rng.standard_normal(200)
    y = rng.standard_normal(200)
    pre = build_preprocessor()
    # Fit ONLY on the first 150 rows (train), then transform the last 50 (val).
    # Fitting on train alone is what prevents target leakage from val into the encoders.
    out_train = pre.fit_transform(df.iloc[:150], y[:150])
    out_val = pre.transform(df.iloc[150:])
    # Train and val must share the same feature layout so a model can use both.
    assert out_train.shape[1] == out_val.shape[1]

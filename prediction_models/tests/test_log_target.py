"""Tests for the log-target wrapper."""

import numpy as np

# `mae` (mean abs error) is imported but not used here; `mape` (mean abs % error)
# is the metric these tests actually exercise.
from ports_dfl.metrics.regression import mae, mape
from ports_dfl.models.linear import LinearRegressor
from ports_dfl.models.log_target import LogTargetWrapper


# pytest finds and runs `test_*` functions automatically. `first_fold_arrays` is a
# fixture (see conftest.py): requesting it by name hands us the preprocessed CV-fold
# arrays. `-> None` is just a type hint (the test returns nothing).
def test_predictions_are_positive(first_fold_arrays) -> None:
    """exp(·) should make all predictions finite and approximately positive."""
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    # Wrap a linear model so it trains on log(target) and exponentiates on predict.
    model = LogTargetWrapper(LinearRegressor(input_dim=n_features, max_epochs=4))
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    # `assert` fails the test if False. First: no NaN/inf got through the exp/clip path.
    assert np.isfinite(preds).all()
    # Predictions are exp(out) - offset, so could be slightly negative when
    # output is near log(offset). Threshold relative to offset.
    # `(preds > -offset - eps).all()` => every prediction sits at or above the
    # theoretical floor (-offset), allowing a tiny 1e-3 float slack.
    assert (preds > -model.offset - 1e-3).all()


def test_log_target_improves_or_matches_mape(first_fold_arrays) -> None:
    """On skewed regression, log-wrapped model usually has better MAPE.

    We don't insist on improvement (data is small here) — just that the
    wrapper trains and produces reasonable predictions.
    """
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    # Baseline: a plain linear model trained directly on the raw target.
    base = LinearRegressor(input_dim=n_features, max_epochs=80, lr=1e-2)
    base.fit(X_train, y_train, X_val, y_val)
    base_mape = mape(y_val, base.predict(X_val))

    # Same model, but wrapped so it learns log(target). Use a fresh inner model
    # so the two runs don't share trained weights.
    log_base = LinearRegressor(input_dim=n_features, max_epochs=80, lr=1e-2)
    wrapped = LogTargetWrapper(log_base)
    wrapped.fit(X_train, y_train, X_val, y_val)
    log_mape = mape(y_val, wrapped.predict(X_val))

    # Both produce finite MAPE
    # Per the docstring, we deliberately do NOT require the wrapper to win on this
    # small data; we only assert both pipelines ran and returned a usable (finite) MAPE.
    assert np.isfinite(base_mape)
    assert np.isfinite(log_mape)


# `tmp_path` is a pytest built-in fixture: a fresh temp directory (a pathlib.Path)
# that pytest creates and deletes for us, so the test never touches real files.
def test_save_load_roundtrip(first_fold_arrays, tmp_path) -> None:
    # Verifies the wrapper survives a save/load cycle: predictions must be unchanged.
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    model = LogTargetWrapper(LinearRegressor(input_dim=n_features, max_epochs=4))
    model.fit(X_train, y_train, X_val, y_val)
    preds_before = model.predict(X_val)

    path = tmp_path / "logwrap.pkl"  # pathlib `/` joins the dir and filename
    model.save(path)
    # LogTargetWrapper.load() restores weights into an EXISTING inner model rather
    # than constructing one, so we must hand it a fresh inner of the right type first.
    new_inner = LinearRegressor(input_dim=n_features)
    restored = LogTargetWrapper(new_inner).load(path)
    preds_after = restored.predict(X_val)
    # Float-tolerant equality (exact == is unsafe for floats). rtol=1e-4 = within
    # 0.01% relative difference, i.e. effectively identical after the round-trip.
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-4)

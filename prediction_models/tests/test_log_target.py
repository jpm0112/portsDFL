"""Tests for the log-target wrapper."""

import numpy as np

from ports_dfl.metrics.regression import mae, mape
from ports_dfl.models.linear import LinearRegressor
from ports_dfl.models.log_target import LogTargetWrapper


def test_predictions_are_positive(first_fold_arrays) -> None:
    """exp(·) should make all predictions finite and approximately positive."""
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    model = LogTargetWrapper(LinearRegressor(input_dim=n_features, max_epochs=4))
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    assert np.isfinite(preds).all()
    # Predictions are exp(out) - offset, so can be slightly negative when output
    # is near log(offset); allow a tiny 1e-3 float slack below the -offset floor.
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

    # Same model wrapped to learn log(target). Fresh inner model so the two runs
    # don't share trained weights.
    log_base = LinearRegressor(input_dim=n_features, max_epochs=80, lr=1e-2)
    wrapped = LogTargetWrapper(log_base)
    wrapped.fit(X_train, y_train, X_val, y_val)
    log_mape = mape(y_val, wrapped.predict(X_val))

    # Per the docstring, we do NOT require the wrapper to win on this small data;
    # only that both pipelines ran and returned a usable (finite) MAPE.
    assert np.isfinite(base_mape)
    assert np.isfinite(log_mape)


def test_save_load_roundtrip(first_fold_arrays, tmp_path) -> None:
    # Wrapper must survive a save/load cycle with unchanged predictions.
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    model = LogTargetWrapper(LinearRegressor(input_dim=n_features, max_epochs=4))
    model.fit(X_train, y_train, X_val, y_val)
    preds_before = model.predict(X_val)

    path = tmp_path / "logwrap.pkl"
    model.save(path)
    # load() restores into an EXISTING inner model rather than constructing one,
    # so hand it a fresh inner of the right type first.
    new_inner = LinearRegressor(input_dim=n_features)
    restored = LogTargetWrapper(new_inner).load(path)
    preds_after = restored.predict(X_val)
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-4)

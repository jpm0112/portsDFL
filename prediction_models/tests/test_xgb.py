"""Tests for the XGBoost regressor benchmark wrapper."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def xgb_cls():
    """Import lazily so a missing xgboost only fails the tests that need it."""
    from ports_dfl.models.xgb import XGBoostRegressorModel

    return XGBoostRegressorModel


def test_fit_predict_shape(xgb_cls, tiny_arrays) -> None:
    """XGBoost fits on a tiny set and returns finite 1D predictions."""
    X_train, y_train, X_val, y_val, n_features = tiny_arrays
    model = xgb_cls(input_dim=n_features, n_estimators=20, early_stopping_rounds=5)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    # One value per validation row, i.e. a flat 1D array.
    assert preds.shape == (len(X_val),)
    assert np.all(np.isfinite(preds))


def test_predict_before_fit_raises(xgb_cls) -> None:
    # predict() on an untrained model must error, not return silent garbage.
    model = xgb_cls(input_dim=10)
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((5, 10)))


def test_fit_without_val_skips_early_stopping(xgb_cls, tiny_arrays) -> None:
    """With no validation set, fit still trains and predicts (early stopping off)."""
    X_train, y_train, _, _, n_features = tiny_arrays
    model = xgb_cls(input_dim=n_features, n_estimators=20)
    model.fit(X_train, y_train)
    assert model.predict(X_train).shape == (len(X_train),)


def test_save_load_roundtrip(xgb_cls, first_fold_arrays, tmp_path) -> None:
    """Reloaded model reproduces the same predictions; save/load loses nothing."""
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    model = xgb_cls(input_dim=n_features, n_estimators=20, early_stopping_rounds=5)
    model.fit(X_train, y_train, X_val, y_val)
    preds_before = model.predict(X_val)

    path = tmp_path / "xgb.pkl"
    model.save(path)
    restored = xgb_cls(input_dim=n_features).load(path)
    preds_after = restored.predict(X_val)

    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)

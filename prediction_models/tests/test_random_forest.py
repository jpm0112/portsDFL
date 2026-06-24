"""Tests for the RandomForest regressor benchmark wrapper."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def rf_cls():
    """Import lazily, consistent with the other model test modules."""
    from ports_dfl.models.random_forest import RandomForestRegressorModel

    return RandomForestRegressorModel


def test_fit_predict_shape(rf_cls, tiny_arrays) -> None:
    """RandomForest fits on a tiny set and returns finite 1D predictions."""
    X_train, y_train, X_val, y_val, n_features = tiny_arrays
    model = rf_cls(input_dim=n_features, n_estimators=20)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)
    assert np.all(np.isfinite(preds))


def test_predict_before_fit_raises(rf_cls) -> None:
    # predict() on an untrained model must error, not return silent garbage.
    model = rf_cls(input_dim=10)
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((5, 10)))


def test_fit_ignores_validation_set(rf_cls, tiny_arrays) -> None:
    """RandomForest has no early stopping, so fitting without a val set works."""
    X_train, y_train, _, _, n_features = tiny_arrays
    model = rf_cls(input_dim=n_features, n_estimators=20)
    model.fit(X_train, y_train)
    assert model.predict(X_train).shape == (len(X_train),)


def test_save_load_roundtrip(rf_cls, first_fold_arrays, tmp_path) -> None:
    """Reloaded model reproduces the same predictions; save/load loses nothing."""
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    model = rf_cls(input_dim=n_features, n_estimators=20)
    model.fit(X_train, y_train, X_val, y_val)
    preds_before = model.predict(X_val)

    path = tmp_path / "rf.pkl"
    model.save(path)
    restored = rf_cls(input_dim=n_features).load(path)
    preds_after = restored.predict(X_val)

    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)

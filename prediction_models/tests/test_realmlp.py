"""Tests for the RealMLP wrapper around pytabkit."""

import warnings

import numpy as np
import pytest

from ports_dfl.metrics.regression import mae

# pytabkit is heavy and noisy; suppress its UserWarnings to keep output clean.
warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="module")
def realmlp_cls():
    """Import RealMLP lazily so other tests don't pay pytabkit's import cost."""
    # Import inside the fixture so a missing pytabkit only fails tests that need
    # it; the rest of the module can still be collected.
    from ports_dfl.models.realmlp import RealMLP

    return RealMLP


def test_fit_predict_shape(realmlp_cls, tiny_arrays) -> None:
    """RealMLP fits on a tiny set and returns 1D predictions."""
    X_train, y_train, X_val, y_val, n_features = tiny_arrays
    model = realmlp_cls(input_dim=n_features, n_epochs=4)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    # One value per validation row, i.e. a flat 1D array.
    assert preds.shape == (len(X_val),)
    # Non-finite predictions would signal the model diverged during training.
    assert np.all(np.isfinite(preds))


def test_predict_before_fit_raises(realmlp_cls) -> None:
    # predict() on an untrained model must error, not return silent garbage.
    model = realmlp_cls(input_dim=10)
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((5, 10)))


# @pytest.mark.slow: trains on the full first fold, filterable via -m "not slow".
@pytest.mark.slow
def test_beats_global_mean_floor(realmlp_cls, first_fold_arrays) -> None:
    """RealMLP with stock defaults should clearly beat the global-mean floor (~22h)."""
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    model = realmlp_cls(input_dim=n_features, n_epochs=32)
    model.fit(X_train, y_train, X_val, y_val)
    val_mae = mae(y_val, model.predict(X_val))
    # Predicting the global mean scores ~22 MAE; < 18 confirms real signal.
    assert val_mae < 18.0


def test_save_load_roundtrip(realmlp_cls, first_fold_arrays, tmp_path) -> None:
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    model = realmlp_cls(input_dim=n_features, n_epochs=4)
    model.fit(X_train, y_train, X_val, y_val)
    preds_before = model.predict(X_val)

    path = tmp_path / "realmlp.pkl"
    model.save(path)
    restored = realmlp_cls(input_dim=n_features).load(path)
    preds_after = restored.predict(X_val)

    # Reloaded model must reproduce the same predictions; save/load loses nothing.
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)

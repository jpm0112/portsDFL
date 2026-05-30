"""Tests for the RealMLP wrapper around pytabkit."""

import warnings

import numpy as np
import pytest

from ports_dfl.metrics.regression import mae

# pytabkit is heavy and noisy; only import once and suppress its info log.
# filterwarnings("ignore", ...) hides warnings of the given category so they
# don't clutter the test output (it does NOT affect pass/fail).
warnings.filterwarnings("ignore", category=UserWarning)


# @pytest.fixture marks a reusable setup routine. A test "requests" it simply by
# listing the fixture's name as a function parameter, and pytest calls this
# function and passes the return value in. scope="module" means it runs once for
# this whole file and the result is cached/shared across its tests.
@pytest.fixture(scope="module")
def realmlp_cls():
    """Import RealMLP lazily so other tests don't pay pytabkit's import cost."""
    # Import is inside the fixture (not at the top of the file) so that if the
    # heavy pytabkit dependency is missing, only tests that need it fail to set
    # up — the rest of the module can still be collected.
    from ports_dfl.models.realmlp import RealMLP

    return RealMLP  # returns the class itself, not an instance


# pytest auto-discovers any function named test_* and runs it as a test.
# The "-> None" is a type hint saying this function returns nothing; it's just
# documentation and does not affect behavior.
# This test follows arrange-act-assert: set up inputs, run the code, then check.
def test_fit_predict_shape(realmlp_cls, tiny_arrays) -> None:
    """RealMLP fits on a tiny set and returns 1D predictions."""
    # Arrange: unpack the tiny_arrays fixture (a tuple) into named variables.
    X_train, y_train, X_val, y_val, n_features = tiny_arrays
    # Act: build the model and train it (only 4 epochs to keep the test fast).
    model = realmlp_cls(input_dim=n_features, n_epochs=4)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    # Assert: a bare `assert` fails the test if the expression is False.
    # Predictions must be one value per validation row, i.e. a flat 1D array.
    assert preds.shape == (len(X_val),)
    # And every prediction must be a real number (no NaN / inf), which would
    # signal the model diverged during training.
    assert np.all(np.isfinite(preds))


def test_predict_before_fit_raises(realmlp_cls) -> None:
    # Calling predict() on an untrained model should be an error, not silent
    # garbage. pytest.raises is a context manager that passes only if the code
    # inside the `with` block raises the given exception type (here RuntimeError);
    # if no error is raised, the test fails.
    model = realmlp_cls(input_dim=10)
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((5, 10)))


# @pytest.mark.slow tags this test so it can be selected/skipped as a group
# (e.g. `pytest -m slow` or `pytest -m "not slow"`). The docstring notes it
# trains on the full first fold and takes a long time (~22h).
@pytest.mark.slow
def test_beats_global_mean_floor(realmlp_cls, first_fold_arrays) -> None:
    """RealMLP with stock defaults should clearly beat the global-mean floor (~22h)."""
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    model = realmlp_cls(input_dim=n_features, n_epochs=32)
    model.fit(X_train, y_train, X_val, y_val)
    val_mae = mae(y_val, model.predict(X_val))
    # A model that just predicts the global mean scores ~22 MAE; requiring < 18
    # checks the model actually learned something useful, not just the average.
    assert val_mae < 18.0


# tmp_path is a built-in pytest fixture: a unique temporary directory (a
# pathlib.Path) that pytest creates and cleans up automatically. Great for
# writing files during a test without touching the real filesystem.
def test_save_load_roundtrip(realmlp_cls, first_fold_arrays, tmp_path) -> None:
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    model = realmlp_cls(input_dim=n_features, n_epochs=4)
    model.fit(X_train, y_train, X_val, y_val)
    preds_before = model.predict(X_val)

    # Save the trained model to disk, then load it into a fresh instance.
    path = tmp_path / "realmlp.pkl"  # `/` joins paths on a pathlib.Path
    model.save(path)
    restored = realmlp_cls(input_dim=n_features).load(path)
    preds_after = restored.predict(X_val)

    # The reloaded model must reproduce the same predictions. assert_allclose is
    # a float-tolerant equality check (rtol = relative tolerance), used because
    # exact == on floats is fragile. This verifies save/load loses nothing.
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)

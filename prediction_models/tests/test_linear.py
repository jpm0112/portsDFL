"""Tests for the LinearRegressor (Ridge) PyTorch model."""

import numpy as np
import pytest
import torch
# `import X as Y` gives a shorter local name; here sklearn's Ridge is renamed so
# it doesn't clash with our own model and reads clearly as the reference impl.
from sklearn.linear_model import Ridge as SklearnRidge

from ports_dfl.metrics.regression import mae
from ports_dfl.models.linear import LinearRegressor


# pytest auto-discovers any function named `test_*` and runs it as a test.
# `tiny_arrays` is a pytest *fixture* (defined in conftest.py): a reusable piece
# of setup. A test "requests" a fixture simply by listing its name as a parameter;
# pytest runs the fixture and passes its return value in. `-> None` is a type hint
# saying the function returns nothing (tests report pass/fail via assertions).
def test_forward_pass_shape(tiny_arrays) -> None:
    # ARRANGE: unpack the fixture's tuple. `_` is the throwaway name for values we
    # don't need here (the val arrays); we only want the train data + feature count.
    X_train, y_train, _, _, n_features = tiny_arrays
    model = LinearRegressor(input_dim=n_features, max_epochs=1)
    # ACT: run a raw forward pass through the underlying torch layer. We convert the
    # numpy array to a float32 tensor first because torch layers operate on tensors.
    out = model.module(torch.as_tensor(X_train, dtype=torch.float32))
    # ASSERT: a bare `assert` makes the test FAIL if the expression is False.
    # The layer should emit one scalar per input row -> shape (n_rows, 1).
    assert out.shape == (len(X_train), 1)
    # No NaN/inf in the output (a fresh, untrained layer should still be numerically sane).
    assert torch.isfinite(out).all()


def test_backward_pass_produces_gradients(tiny_arrays) -> None:
    # Checks the model is actually trainable: a backward pass must fill in
    # gradients for every parameter (otherwise the optimizer would have nothing
    # to update and "training" would silently do nothing).
    X_train, y_train, _, _, n_features = tiny_arrays
    model = LinearRegressor(input_dim=n_features, max_epochs=1)
    X_t = torch.as_tensor(X_train, dtype=torch.float32)
    # reshape(-1, 1) turns a flat (N,) target vector into an (N, 1) column to match
    # the model's (N, 1) output. -1 means "infer this dimension from the size".
    y_t = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    # ACT: forward pass, compute MSE loss, then backprop to populate `.grad`.
    pred = model.module(X_t)
    loss = torch.nn.functional.mse_loss(pred, y_t)
    loss.backward()  # torch walks the graph backward and fills each param's .grad
    # ASSERT: every trainable parameter must have received a finite gradient.
    for p in model.module.parameters():
        assert p.grad is not None  # None would mean this param got no gradient at all
        assert torch.isfinite(p.grad).all()  # and the gradient must not be NaN/inf


def test_overfit_tiny_dataset(tiny_arrays) -> None:
    """A linear model should drive train MAE clearly below the predict-the-mean
    baseline on a tiny set (within ~30% of the closed-form linear-regression
    optimum, which is ~11h on this 64-row slice with 31 features)."""
    X_train, y_train, _, _, n_features = tiny_arrays
    model = LinearRegressor(
        input_dim=n_features,
        lr=1e-2,
        weight_decay=0.0,
        batch_size=64,
        max_epochs=400,
        patience=400,
    )
    # Train and evaluate on the SAME tiny set on purpose: this is an over-fit
    # smoke test, so train==val is intentional (we want to see it learn, not generalize).
    model.fit(X_train, y_train, X_train, y_train)
    train_mae_model = mae(y_train, model.predict(X_train))
    # Floor: predicting the training mean
    # np.full_like(arr, v) makes an array the same shape/dtype as `arr` filled with v.
    # So this is the MAE of the trivial "always guess the average" predictor.
    train_mae_floor = mae(y_train, np.full_like(y_train, y_train.mean()))
    # ASSERT: a real linear fit must beat that trivial baseline by a clear margin
    # (here, at least 15% lower MAE). The second arg to `assert` is the message
    # printed on failure; the f"..." is an f-string and {x:.2f} formats x to 2 decimals.
    assert train_mae_model < 0.85 * train_mae_floor, (
        f"Linear barely beats mean baseline: {train_mae_model:.2f} "
        f"vs floor {train_mae_floor:.2f}"
    )


# @pytest.mark.slow tags this test with a custom "slow" marker. It does NOT skip
# the test by itself; it just lets you select/deselect it from the command line
# (e.g. run only fast tests with `-m "not slow"`). Useful because this one trains
# on the full first fold and is comparatively slow.
@pytest.mark.slow
def test_one_epoch_beats_global_mean(first_fold_arrays) -> None:
    """After modest training, val MAE should beat the global-mean floor (~22h)."""
    # Uses the full first CV fold (not the tiny slice), so train/val are distinct.
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    model = LinearRegressor(
        input_dim=n_features,
        lr=1e-2,
        weight_decay=1e-3,
        batch_size=256,
        max_epochs=20,
        patience=20,
    )
    model.fit(X_train, y_train, X_val, y_val)
    # Evaluate on held-out validation rows. The 22.0 ceiling is the global-mean
    # baseline MAE in hours (see docstring); beating it proves the model learned signal.
    val_mae = mae(y_val, model.predict(X_val))
    assert val_mae < 22.0


# `tmp_path` is a built-in pytest fixture: a unique temporary directory (a
# pathlib.Path) that pytest creates for this test and cleans up afterward. Using
# it keeps the test from leaving files behind or clobbering anything real.
def test_save_load_roundtrip(first_fold_arrays, tmp_path) -> None:
    # Verifies save()->load() preserves the model exactly: predictions before and
    # after a disk round-trip must match.
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    model = LinearRegressor(input_dim=n_features, max_epochs=5, patience=5)
    model.fit(X_train, y_train, X_val, y_val)
    preds_before = model.predict(X_val)  # baseline predictions to compare against

    # `tmp_path / "linear.pt"` uses pathlib's `/` operator to join a path segment.
    path = tmp_path / "linear.pt"
    model.save(path)
    # Build a brand-new model and load the saved weights into it (load() returns
    # self, so we can chain .load(path) right onto the constructor).
    restored = LinearRegressor(input_dim=n_features).load(path)
    preds_after = restored.predict(X_val)

    # np.testing.assert_allclose checks two float arrays are equal within a
    # tolerance (exact == is unreliable for floats). rtol=1e-5 = allow 0.001%
    # relative difference; here we expect a near-perfect match after reloading.
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)


def test_pytorch_ridge_agrees_with_sklearn(first_fold_arrays) -> None:
    """PyTorch Ridge with strong regularization should match sklearn closely."""
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays

    # Train PyTorch Ridge for many epochs with strong regularization
    model = LinearRegressor(
        input_dim=n_features,
        lr=5e-2,
        weight_decay=1e-3,
        batch_size=256,
        max_epochs=300,
        patience=300,
    )
    model.fit(X_train, y_train, X_val, y_val)
    pt_preds = model.predict(X_val)  # our PyTorch model's predictions
    pt_mae = mae(y_val, pt_preds)

    # Closed-form sklearn Ridge with equivalent alpha (alpha = wd * n_train)
    # The two libraries scale their L2 penalty differently, so we convert our
    # per-sample weight_decay into sklearn's total-penalty `alpha`. max(..., 1e-6)
    # keeps alpha strictly positive even if weight_decay were 0.
    alpha = max(model.weight_decay * len(X_train), 1e-6)
    # sklearn's Ridge has an exact closed-form solution -> our reference "truth".
    sk = SklearnRidge(alpha=alpha, random_state=42).fit(X_train, y_train)
    sk_preds = sk.predict(X_val)
    sk_mae = mae(y_val, sk_preds)

    # Loose tolerance because Adam + minibatch != closed-form least squares
    # ASSERT: relative gap between the two MAEs stays under 10%. We compute
    # |a-b| / max(b, 1e-6); dividing by max(...,1e-6) just avoids divide-by-zero.
    assert abs(pt_mae - sk_mae) / max(sk_mae, 1e-6) < 0.10, (
        f"PyTorch MAE {pt_mae:.3f} diverges >10% from sklearn MAE {sk_mae:.3f}"
    )


def test_uses_cuda_when_available(tiny_arrays) -> None:
    """If CUDA is available, fitted parameters should live on cuda after fit."""
    # pytest.skip(reason) aborts THIS test as "skipped" (not passed, not failed).
    # On a machine without a GPU there's nothing to check, so we bail out cleanly.
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    X_train, y_train, _, _, n_features = tiny_arrays
    model = LinearRegressor(input_dim=n_features, max_epochs=2)
    model.fit(X_train, y_train, X_train, y_train)
    # `next(iter)` pulls the first item from .parameters() (which yields one tensor
    # at a time). We check that first weight tensor ended up on the GPU after fit().
    assert next(model.module.parameters()).device.type == "cuda"

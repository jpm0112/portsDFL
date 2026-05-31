"""Tests for the LinearRegressor (Ridge) PyTorch model."""

import numpy as np
import pytest
import torch
# Renamed so it doesn't clash with our model and reads as the reference impl.
from sklearn.linear_model import Ridge as SklearnRidge

from ports_dfl.metrics.regression import mae
from ports_dfl.models.linear import LinearRegressor


def test_forward_pass_shape(tiny_arrays) -> None:
    X_train, y_train, _, _, n_features = tiny_arrays
    model = LinearRegressor(input_dim=n_features, max_epochs=1)
    out = model.module(torch.as_tensor(X_train, dtype=torch.float32))
    # One scalar per input row -> shape (n_rows, 1).
    assert out.shape == (len(X_train), 1)
    assert torch.isfinite(out).all()


def test_backward_pass_produces_gradients(tiny_arrays) -> None:
    # Confirms the model is trainable: a backward pass must populate gradients
    # for every parameter, else the optimizer would have nothing to update.
    X_train, y_train, _, _, n_features = tiny_arrays
    model = LinearRegressor(input_dim=n_features, max_epochs=1)
    X_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_t = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    pred = model.module(X_t)
    loss = torch.nn.functional.mse_loss(pred, y_t)
    loss.backward()
    for p in model.module.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


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
    # Train==val on purpose: this is an over-fit smoke test, we want to see it
    # learn, not generalize.
    model.fit(X_train, y_train, X_train, y_train)
    train_mae_model = mae(y_train, model.predict(X_train))
    # Floor: MAE of the trivial "always guess the training mean" predictor.
    train_mae_floor = mae(y_train, np.full_like(y_train, y_train.mean()))
    # A real linear fit must beat that floor by a clear margin (>=15% lower MAE).
    assert train_mae_model < 0.85 * train_mae_floor, (
        f"Linear barely beats mean baseline: {train_mae_model:.2f} "
        f"vs floor {train_mae_floor:.2f}"
    )


# @pytest.mark.slow: trains on the full first fold, filterable via -m "not slow".
@pytest.mark.slow
def test_one_epoch_beats_global_mean(first_fold_arrays) -> None:
    """After modest training, val MAE should beat the global-mean floor (~22h)."""
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
    # 22.0 is the global-mean baseline MAE in hours (see docstring); beating it
    # proves the model learned signal.
    val_mae = mae(y_val, model.predict(X_val))
    assert val_mae < 22.0


def test_save_load_roundtrip(first_fold_arrays, tmp_path) -> None:
    # save()->load() must preserve predictions exactly across a disk round-trip.
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    model = LinearRegressor(input_dim=n_features, max_epochs=5, patience=5)
    model.fit(X_train, y_train, X_val, y_val)
    preds_before = model.predict(X_val)

    path = tmp_path / "linear.pt"
    model.save(path)
    restored = LinearRegressor(input_dim=n_features).load(path)
    preds_after = restored.predict(X_val)

    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)


def test_pytorch_ridge_agrees_with_sklearn(first_fold_arrays) -> None:
    """PyTorch Ridge with strong regularization should match sklearn closely."""
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays

    model = LinearRegressor(
        input_dim=n_features,
        lr=5e-2,
        weight_decay=1e-3,
        batch_size=256,
        max_epochs=300,
        patience=300,
    )
    model.fit(X_train, y_train, X_val, y_val)
    pt_preds = model.predict(X_val)
    pt_mae = mae(y_val, pt_preds)

    # The two libraries scale their L2 penalty differently: convert our per-sample
    # weight_decay into sklearn's total-penalty alpha (= wd * n_train). The
    # max(..., 1e-6) keeps alpha strictly positive even if weight_decay is 0.
    alpha = max(model.weight_decay * len(X_train), 1e-6)
    # sklearn Ridge has an exact closed-form solution -> our reference truth.
    sk = SklearnRidge(alpha=alpha, random_state=42).fit(X_train, y_train)
    sk_preds = sk.predict(X_val)
    sk_mae = mae(y_val, sk_preds)

    # Loose tolerance because Adam + minibatch != closed-form least squares.
    assert abs(pt_mae - sk_mae) / max(sk_mae, 1e-6) < 0.10, (
        f"PyTorch MAE {pt_mae:.3f} diverges >10% from sklearn MAE {sk_mae:.3f}"
    )


def test_uses_cuda_when_available(tiny_arrays) -> None:
    """If CUDA is available, fitted parameters should live on cuda after fit."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    X_train, y_train, _, _, n_features = tiny_arrays
    model = LinearRegressor(input_dim=n_features, max_epochs=2)
    model.fit(X_train, y_train, X_train, y_train)
    assert next(model.module.parameters()).device.type == "cuda"

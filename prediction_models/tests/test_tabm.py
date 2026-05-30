"""Tests for TabM (parameter-efficient ensemble of MLPs)."""

import numpy as np
import pytest
import torch

from ports_dfl.metrics.regression import mae


# @pytest.fixture is a reusable setup; a test gets it by naming it as a parameter.
# scope="module" runs it once per file and caches the result. The TabM import is
# done lazily inside so a missing torch/TabM dependency only breaks tests that
# actually use it, not the whole module's collection.
@pytest.fixture(scope="module")
def tabm_cls():
    from ports_dfl.models.tabm import TabM

    return TabM  # hand back the class itself for tests to instantiate


# pytest runs every function named test_* automatically. "-> None" is just a
# type hint (returns nothing). Structure is arrange (set up) - act (run) - assert.
def test_forward_pass_shape(tabm_cls, tiny_arrays) -> None:
    # We only need the training features here; `_` is the conventional name for
    # values we unpack but intentionally ignore.
    X_train, y_train, _, _, n_features = tiny_arrays
    model = tabm_cls(input_dim=n_features, max_epochs=1, k_ensemble=4, hidden_dim=64, depth=2)
    # Convert the numpy array into a float32 torch tensor the model can consume.
    X_t = torch.as_tensor(X_train, dtype=torch.float32)
    out = model.module(X_t)  # run the raw network forward (no training)
    # A bare `assert` fails the test if False. Output must be one value per row:
    # shape (n_rows, 1).
    assert out.shape == (len(X_train), 1)
    # Every output must be finite (no NaN/inf) — catches a broken forward pass.
    assert torch.isfinite(out).all()


def test_backward_pass_produces_gradients(tabm_cls, tiny_arrays) -> None:
    X_train, y_train, _, _, n_features = tiny_arrays
    model = tabm_cls(input_dim=n_features, max_epochs=1, k_ensemble=4, hidden_dim=64, depth=2)
    X_t = torch.as_tensor(X_train, dtype=torch.float32)
    # reshape(-1, 1) turns the 1D targets into a column to match the (N, 1) output;
    # -1 means "infer this dimension from the data".
    y_t = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    pred = model.module(X_t)
    loss = torch.nn.functional.mse_loss(pred, y_t)
    loss.backward()  # backpropagate: compute gradients for every parameter
    # This is a generator expression: count parameters that received a gradient.
    # `sum(1 for p in ... if cond)` adds 1 for each parameter whose .grad is set.
    n_grads = sum(1 for p in model.module.parameters() if p.grad is not None)
    # If at least one parameter got a gradient, backprop is wired up correctly
    # (the network is actually trainable, not disconnected from the loss).
    assert n_grads > 0


# @pytest.mark.slow tags this so it can be filtered (e.g. `pytest -m "not slow"`).
# It trains on the full first fold, which is expensive.
@pytest.mark.slow
def test_beats_global_mean_floor(tabm_cls, first_fold_arrays) -> None:
    """TabM with modest training should clearly beat the global-mean floor."""
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    model = tabm_cls(
        input_dim=n_features,
        k_ensemble=8, hidden_dim=192, depth=2,
        max_epochs=32, patience=10,
    )
    model.fit(X_train, y_train, X_val, y_val)
    val_mae = mae(y_val, model.predict(X_val))
    # Predicting the global mean scores ~22 MAE; < 18 proves the model learned.
    assert val_mae < 18.0


# tmp_path is a built-in pytest fixture: a fresh temporary directory (Path) that
# pytest creates and deletes automatically, so the test never touches real files.
def test_save_load_roundtrip(tabm_cls, first_fold_arrays, tmp_path) -> None:
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    model = tabm_cls(
        input_dim=n_features,
        k_ensemble=4, hidden_dim=64, depth=2,
        max_epochs=4, patience=4,
    )
    model.fit(X_train, y_train, X_val, y_val)
    preds_before = model.predict(X_val)

    # Save to disk, then reload into a brand-new instance.
    path = tmp_path / "tabm.pt"
    model.save(path)
    restored = tabm_cls(input_dim=n_features).load(path)
    preds_after = restored.predict(X_val)

    # assert_allclose is float-tolerant equality (rtol = relative tolerance).
    # The reloaded model must reproduce the original predictions, so save/load
    # round-trips the full state. Looser rtol than RealMLP because torch tensors
    # serialize with slightly more float jitter.
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-4)


def test_uses_cuda_when_available(tabm_cls, tiny_arrays) -> None:
    # Skip (don't fail) when there's no GPU — this machine simply can't run it.
    # pytest.skip marks the test as skipped at runtime with the given reason.
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    X_train, y_train, _, _, n_features = tiny_arrays
    model = tabm_cls(input_dim=n_features, k_ensemble=4, hidden_dim=64, depth=2, max_epochs=2)
    model.fit(X_train, y_train, X_train, y_train)
    # After fitting on a CUDA machine, the model's parameters should live on the
    # GPU. next(...parameters()) grabs the first parameter tensor to inspect it.
    assert next(model.module.parameters()).device.type == "cuda"

"""Tests for NODE (neural oblivious decision ensembles)."""

import numpy as np
import pytest
import torch

from ports_dfl.metrics.regression import mae


# @pytest.fixture is a reusable setup; a test "requests" it by listing its name
# as a parameter. scope="module" runs it once per file and caches the result.
# The NODE import is lazy (inside the function) so a missing dependency only
# breaks tests that need it rather than the whole file.
@pytest.fixture(scope="module")
def node_cls():
    from ports_dfl.models.node import NODE

    return NODE  # return the class for tests to instantiate


# pytest auto-runs functions named test_*. "-> None" is a type hint (no return).
# Pattern: arrange (set up data/model), act (run it), assert (check the result).
def test_forward_pass_shape(node_cls, tiny_arrays) -> None:
    # Only the training features are needed; `_` ignores the unpacked values.
    X_train, _, _, _, n_features = tiny_arrays
    model = node_cls(
        input_dim=n_features, n_layers=2, n_trees=32, tree_depth=4, max_epochs=1
    )
    # Wrap the numpy features as a float32 torch tensor for the network.
    X_t = torch.as_tensor(X_train, dtype=torch.float32)
    out = model.module(X_t)  # raw forward pass, no training involved
    # A bare `assert` fails the test if False. One output per input row: (N, 1).
    assert out.shape == (len(X_train), 1)
    # No NaN/inf in the output — guards against a broken forward computation.
    assert torch.isfinite(out).all()


def test_backward_pass_produces_gradients(node_cls, tiny_arrays) -> None:
    X_train, y_train, _, _, n_features = tiny_arrays
    model = node_cls(
        input_dim=n_features, n_layers=2, n_trees=32, tree_depth=4, max_epochs=1
    )
    X_t = torch.as_tensor(X_train, dtype=torch.float32)
    # reshape(-1, 1): make targets a column to match the (N, 1) output; -1 means
    # "infer this size from the array".
    y_t = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    pred = model.module(X_t)
    loss = torch.nn.functional.mse_loss(pred, y_t)
    loss.backward()  # backprop: fill in .grad for each trainable parameter
    # Generator expression counting parameters that actually received a gradient
    # (sum adds 1 per matching parameter).
    n_grads = sum(1 for p in model.module.parameters() if p.grad is not None)
    # >0 means the loss connects to the parameters and the model is trainable.
    assert n_grads > 0


# @pytest.mark.slow tags this for filtering (e.g. `pytest -m "not slow"`); it
# trains on the full first fold and is expensive to run.
@pytest.mark.slow
def test_beats_global_mean_floor(node_cls, first_fold_arrays) -> None:
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    model = node_cls(
        input_dim=n_features,
        n_layers=2, n_trees=128, tree_depth=6,
        max_epochs=32, patience=10,
    )
    model.fit(X_train, y_train, X_val, y_val)
    val_mae = mae(y_val, model.predict(X_val))
    # Predicting the global mean scores ~22 MAE; requiring < 18 confirms the
    # model learned real signal rather than just the average target.
    assert val_mae < 18.0


# tmp_path is a built-in pytest fixture: a unique temp directory (Path) pytest
# creates and cleans up, so the test writes files without touching real paths.
def test_save_load_roundtrip(node_cls, first_fold_arrays, tmp_path) -> None:
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    model = node_cls(
        input_dim=n_features,
        n_layers=2, n_trees=32, tree_depth=4,
        max_epochs=4, patience=4,
    )
    model.fit(X_train, y_train, X_val, y_val)
    preds_before = model.predict(X_val)

    # Persist the trained model, then reload it into a fresh instance.
    path = tmp_path / "node.pt"
    model.save(path)
    restored = node_cls(input_dim=n_features).load(path)
    preds_after = restored.predict(X_val)

    # assert_allclose is float-tolerant equality (rtol = relative tolerance).
    # The reloaded model must give the same predictions, proving save/load
    # preserves the full model state.
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-4)


def test_uses_cuda_when_available(node_cls, tiny_arrays) -> None:
    # No GPU on this machine -> skip rather than fail. pytest.skip stops the test
    # at runtime and marks it "skipped" with the given reason.
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    X_train, y_train, _, _, n_features = tiny_arrays
    model = node_cls(
        input_dim=n_features, n_layers=2, n_trees=32, tree_depth=4, max_epochs=2
    )
    model.fit(X_train, y_train, X_train, y_train)
    # On a CUDA machine the trained model's first parameter tensor should sit on
    # the GPU; next(...parameters()) grabs that first parameter to check.
    assert next(model.module.parameters()).device.type == "cuda"

"""Tests for NODE (neural oblivious decision ensembles)."""

import numpy as np
import pytest
import torch

from ports_dfl.metrics.regression import mae


@pytest.fixture(scope="module")
def node_cls():
    from ports_dfl.models.node import NODE

    return NODE


def test_forward_pass_shape(node_cls, tiny_arrays) -> None:
    X_train, _, _, _, n_features = tiny_arrays
    model = node_cls(
        input_dim=n_features, n_layers=2, n_trees=32, tree_depth=4, max_epochs=1
    )
    X_t = torch.as_tensor(X_train, dtype=torch.float32)
    out = model.module(X_t)
    assert out.shape == (len(X_train), 1)
    assert torch.isfinite(out).all()


def test_backward_pass_produces_gradients(node_cls, tiny_arrays) -> None:
    X_train, y_train, _, _, n_features = tiny_arrays
    model = node_cls(
        input_dim=n_features, n_layers=2, n_trees=32, tree_depth=4, max_epochs=1
    )
    X_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_t = torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    pred = model.module(X_t)
    loss = torch.nn.functional.mse_loss(pred, y_t)
    loss.backward()
    n_grads = sum(1 for p in model.module.parameters() if p.grad is not None)
    assert n_grads > 0


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
    assert val_mae < 18.0


def test_save_load_roundtrip(node_cls, first_fold_arrays, tmp_path) -> None:
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    model = node_cls(
        input_dim=n_features,
        n_layers=2, n_trees=32, tree_depth=4,
        max_epochs=4, patience=4,
    )
    model.fit(X_train, y_train, X_val, y_val)
    preds_before = model.predict(X_val)

    path = tmp_path / "node.pt"
    model.save(path)
    restored = node_cls(input_dim=n_features).load(path)
    preds_after = restored.predict(X_val)

    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-4)


def test_uses_cuda_when_available(node_cls, tiny_arrays) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    X_train, y_train, _, _, n_features = tiny_arrays
    model = node_cls(
        input_dim=n_features, n_layers=2, n_trees=32, tree_depth=4, max_epochs=2
    )
    model.fit(X_train, y_train, X_train, y_train)
    assert next(model.module.parameters()).device.type == "cuda"

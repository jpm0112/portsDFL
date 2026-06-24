"""NODE (Neural Oblivious Decision Ensembles) wrapper.

Tier 4 of the model lineup. NODE (Popov et al., ICLR 2020) stacks
differentiable oblivious decision trees with dense (skip) connections.
Tree inductive bias plus end-to-end gradient flow.

Implementation uses ODST and DenseODSTBlock from pytorch_tabular's NODE module
(maintained re-implementation of Qwicen/node).

Reference: https://arxiv.org/abs/1909.06312
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pytorch_tabular.models.common.layers.soft_trees import (
    entmax15,
    sparsemax,
    sparsemoid,
)
from pytorch_tabular.models.node.architecture_blocks import DenseODSTBlock

from ports_dfl.config import DEVICE, SEED
from ports_dfl.models.base import BaseModel
from ports_dfl.train.pto import TrainConfig, predict_pto, train_pto


# Map config string names to function objects so a function can be picked at runtime.
# Choice functions select between paths down the tree; bin function decides leaf
# membership. The original NODE paper uses entmax/entmoid, but only sparsemoid is
# exposed by pytorch-tabular's port; entmax15 is the default and works with it.
CHOICE_FUNCTIONS = {
    "entmax15": entmax15,
    "sparsemax": sparsemax,
}

BIN_FUNCTIONS = {
    "sparsemoid": sparsemoid,
}


class _NODERegressor(nn.Module):
    """Stack of dense ODT layers with a linear regression head."""

    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        n_trees: int,
        tree_depth: int,
        tree_output_dim: int = 1,
        choice_function: str = "entmax15",
        bin_function: str = "sparsemoid",
        input_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # DenseODSTBlock is the stack of differentiable oblivious decision trees.
        # FIX: forward tree_depth to the block (pytorch_tabular's ODST keyword is
        # `depth`). It was accepted by __init__ but never passed here, so the
        # tree-depth hyperparameter was silently ignored and any tuning sweep over
        # it had no effect. NOTE: changes the network (old checkpoints won't match);
        # confirm the kwarg name against the installed pytorch_tabular version.
        self.block = DenseODSTBlock(
            input_dim=input_dim,
            num_trees=n_trees,
            num_layers=n_layers,
            depth=tree_depth,
            tree_output_dim=tree_output_dim,
            input_dropout=input_dropout,
            flatten_output=True,  # return one flat feature vector per row
            choice_function=CHOICE_FUNCTIONS[choice_function],
            bin_function=BIN_FUNCTIONS[bin_function],
        )
        # The block flattens tree outputs to (batch, n_layers*n_trees*tree_output_dim)
        # so the head's input width must equal that product.
        feat_dim = n_layers * n_trees * tree_output_dim
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(x)
        return self.head(h)


class NODE(BaseModel):
    """BaseModel-compatible NODE regressor."""

    def __init__(
        self,
        input_dim: int,
        n_layers: int = 4,
        n_trees: int = 256,
        tree_depth: int = 6,
        choice_function: str = "entmax15",
        bin_function: str = "sparsemoid",
        input_dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 16,
        grad_clip: float = 1.0,
        use_amp: bool = False,
        seed: int = SEED,
    ) -> None:
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.choice_function = choice_function
        self.bin_function = bin_function
        self.input_dropout = input_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.seed = seed  # per-fold seed -> TrainConfig, so CV folds are independent draws
        self.module = self._build_module()
        self.train_result_ = None

    def _build_module(self) -> _NODERegressor:
        # Used at __init__ and when reloading (so the architecture matches the
        # saved weights before loading them).
        return _NODERegressor(
            input_dim=self.input_dim,
            n_layers=self.n_layers,
            n_trees=self.n_trees,
            tree_depth=self.tree_depth,
            choice_function=self.choice_function,
            bin_function=self.bin_function,
            input_dropout=self.input_dropout,
        )

    def _train_config(self) -> TrainConfig:
        return TrainConfig(
            lr=self.lr,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            patience=self.patience,
            grad_clip=self.grad_clip,
            use_amp=self.use_amp,
            seed=self.seed,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "NODE":
        if X_val is None or y_val is None:
            n_val = max(int(len(X_train) * 0.1), 1)
            # Plain tail split (no shuffle), appropriate for time-ordered data
            # to avoid look-ahead leakage.
            X_val, y_val = X_train[-n_val:], y_train[-n_val:]
            X_train, y_train = X_train[:-n_val], y_train[:-n_val]
        self.train_result_ = train_pto(
            self.module, X_train, y_train, X_val, y_val, self._train_config()
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Bigger batch at inference (no gradients stored) for speed.
        return predict_pto(self.module, X, batch_size=self.batch_size * 4)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"init_kwargs": self._init_kwargs(), "state_dict": self.module.state_dict()},
            path,
        )

    def load(self, path: Path | str) -> "NODE":
        # weights_only=False is needed because we also stored the plain-dict
        # init_kwargs (not just tensors).
        ckpt = torch.load(Path(path), map_location=DEVICE, weights_only=False)
        for k, v in ckpt["init_kwargs"].items():
            setattr(self, k, v)
        # Rebuild from restored hyperparameters, THEN load weights (architecture
        # must match the saved state_dict).
        self.module = self._build_module()
        self.module.load_state_dict(ckpt["state_dict"])
        self.module.to(DEVICE)
        return self

    def _init_kwargs(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "n_layers": self.n_layers,
            "n_trees": self.n_trees,
            "tree_depth": self.tree_depth,
            "choice_function": self.choice_function,
            "bin_function": self.bin_function,
            "input_dropout": self.input_dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "grad_clip": self.grad_clip,
            "use_amp": self.use_amp,
        }

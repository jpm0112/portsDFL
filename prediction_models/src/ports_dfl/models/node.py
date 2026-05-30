"""NODE (Neural Oblivious Decision Ensembles) wrapper.

Tier 4 of the model lineup. NODE (Popov et al., ICLR 2020) stacks
differentiable oblivious decision trees with dense (skip) connections.
Tree inductive bias plus end-to-end gradient flow.

Implementation uses ODST and DenseODSTBlock from pytorch_tabular's NODE module
(maintained re-implementation of Qwicen/node).

Reference: https://arxiv.org/abs/1909.06312
"""

# `Path` is an object-oriented file-path type from the standard library.
from pathlib import Path

import numpy as np
import torch

# `torch.nn` holds neural-network building blocks (layers, etc.). The `as nn`
# part gives the import a shorter nickname so we can write `nn.Linear`.
import torch.nn as nn

# These come from pytorch_tabular's port of the NODE architecture. The
# parentheses let us import several names across multiple lines.
from pytorch_tabular.models.common.layers.soft_trees import (
    entmax15,
    sparsemax,
    sparsemoid,
)
from pytorch_tabular.models.node.architecture_blocks import DenseODSTBlock

from ports_dfl.config import DEVICE
from ports_dfl.models.base import BaseModel
from ports_dfl.train.pto import TrainConfig, predict_pto, train_pto


# These are plain dictionaries mapping a human-readable string name (the key)
# to the actual function object (the value). Storing the function itself as a
# value lets us pick a function at runtime from a config string, e.g.
# CHOICE_FUNCTIONS["entmax15"] returns the entmax15 function.
# Choice functions select between paths down the tree. Bin function decides
# leaf membership. The original NODE paper uses entmax/entmoid, but only
# sparsemoid is exposed by pytorch-tabular's port; entmax15 is the default
# choice function and works with sparsemoid in practice.
CHOICE_FUNCTIONS = {
    "entmax15": entmax15,
    "sparsemax": sparsemax,
}

BIN_FUNCTIONS = {
    "sparsemoid": sparsemoid,
}


# `class _NODERegressor(nn.Module)` means this class INHERITS from nn.Module,
# the base class for every PyTorch model/layer. The leading underscore in the
# name is a convention meaning "internal/private — not part of the public API".
class _NODERegressor(nn.Module):
    """Stack of dense ODT layers with a linear regression head."""

    # `__init__` is the constructor — it runs when you create the object.
    # `self` is the object being built. Type hints like `input_dim: int` and
    # default values like `tree_output_dim: int = 1` document/guard the inputs.
    # `-> None` says this method returns nothing.
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
        # `super().__init__()` calls nn.Module's own constructor. PyTorch needs
        # this to register parameters/sub-modules; skipping it breaks the model.
        super().__init__()
        # The DenseODSTBlock is the actual stack of differentiable oblivious
        # decision trees. Assigning it to `self.block` registers it as a
        # sub-module so its weights get tracked and trained automatically.
        # REVIEW NOTE: `tree_depth` is accepted by this constructor (and by
        # NODE.__init__) but is NOT forwarded to DenseODSTBlock below, so the
        # tree-depth hyperparameter is silently ignored and the block always
        # uses its library default depth. To actually control depth, pass it
        # through (the pytorch_tabular ODST API uses the keyword `depth=`),
        # e.g. add `depth=tree_depth,` to this call. Left unchanged here
        # because it alters the network architecture/behavior.
        self.block = DenseODSTBlock(
            input_dim=input_dim,
            num_trees=n_trees,
            num_layers=n_layers,
            tree_output_dim=tree_output_dim,
            input_dropout=input_dropout,
            flatten_output=True,  # return one flat feature vector per row
            # Look up the function object from the dicts using the string name.
            choice_function=CHOICE_FUNCTIONS[choice_function],
            bin_function=BIN_FUNCTIONS[bin_function],
        )
        # The block flattens tree outputs to (batch, n_layers*n_trees*tree_output_dim)
        # so the head's input width must equal that product.
        feat_dim = n_layers * n_trees * tree_output_dim
        # A single Linear layer maps the flattened tree features down to one
        # scalar regression output per row.
        self.head = nn.Linear(feat_dim, 1)

    # `forward` defines what happens when the model is called on input `x`.
    # PyTorch runs this automatically when you do `model(x)`.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(x)  # run the tree ensemble -> flattened features
        return self.head(h)  # project features to the final prediction


# NODE inherits from BaseModel (the project's shared interface), so the
# training/tuning/eval code can treat every model the same way.
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
    ) -> None:
        # Store every hyperparameter on `self` so we can rebuild the module
        # later (e.g. after load) and report the exact config via _init_kwargs.
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
        # Build the actual PyTorch module now so it's ready to train.
        self.module = self._build_module()
        # Trailing-underscore name is a scikit-learn convention meaning "filled
        # in after fit()". None until fit() runs and stores the TrainResult.
        self.train_result_ = None

    def _build_module(self) -> _NODERegressor:
        # Constructs a fresh network from the stored hyperparameters. Used both
        # at __init__ and when reloading a saved model (so the architecture
        # matches the saved weights before loading them).
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
        )

    # `X_val: np.ndarray | None = None` — the `| None` type hint means this
    # argument may be a numpy array OR None; default is None (validation
    # set is optional). The return type `"NODE"` is quoted because the class
    # isn't fully defined yet at this point in the file (a forward reference).
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "NODE":
        # If no validation set was provided, carve one out of the tail of the
        # training data so early stopping still has something to monitor.
        if X_val is None or y_val is None:
            # At least 1 row; ~10% of the training rows otherwise.
            n_val = max(int(len(X_train) * 0.1), 1)
            # `arr[-n_val:]` = last n_val rows; `arr[:-n_val]` = all but the
            # last n_val rows. Note: this is a plain tail split (no shuffle),
            # appropriate for time-ordered data to avoid look-ahead leakage.
            X_val, y_val = X_train[-n_val:], y_train[-n_val:]
            X_train, y_train = X_train[:-n_val], y_train[:-n_val]
        # Hand off to the shared training loop; it trains in place and returns
        # a TrainResult trace we stash for later inspection.
        self.train_result_ = train_pto(
            self.module, X_train, y_train, X_val, y_val, self._train_config()
        )
        return self  # return self so calls can be chained, e.g. model.fit(...).predict(...)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Larger batch for inference (no gradients) — 4x the training batch.
        return predict_pto(self.module, X, batch_size=self.batch_size * 4)

    def save(self, path: Path | str) -> None:
        # Accept either a Path or a string; wrap in Path so the rest works.
        path = Path(path)
        # Create parent folders if missing; `exist_ok=True` avoids erroring
        # when they already exist.
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save BOTH the hyperparameters (to rebuild the architecture) and the
        # learned weights (state_dict). The argument is a plain dict.
        torch.save(
            {"init_kwargs": self._init_kwargs(), "state_dict": self.module.state_dict()},
            path,
        )

    def load(self, path: Path | str) -> "NODE":
        # `map_location=DEVICE` loads tensors onto our device (CPU/GPU) even if
        # they were saved from a different one. weights_only=False is needed
        # because we also stored the plain-dict init_kwargs (not just tensors).
        ckpt = torch.load(Path(path), map_location=DEVICE, weights_only=False)
        # Restore every saved hyperparameter onto self. `.items()` yields
        # (key, value) pairs; `setattr(self, k, v)` is the dynamic form of
        # `self.k = v` where the attribute name comes from a variable.
        for k, v in ckpt["init_kwargs"].items():
            setattr(self, k, v)
        # Rebuild the network from those restored hyperparameters, THEN load the
        # saved weights into it (architecture must match the saved state_dict).
        self.module = self._build_module()
        self.module.load_state_dict(ckpt["state_dict"])
        self.module.to(DEVICE)  # move weights onto the active device
        return self

    # Returns a dict of every constructor argument, used by save() and to
    # reconstruct the model. `-> dict` is the return-type hint.
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

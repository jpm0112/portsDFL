"""TabM (parameter-efficient ensemble of MLPs) wrapper.

Tier 3 of the model lineup. TabM (Gorishniy et al., ICLR 2025) achieves
ensemble-level performance with a single shared backbone via the BatchEnsemble
trick. Top-ranked on TALENT/TabArena 2025 benchmarks.

Reference: https://github.com/yandex-research/tabm
PyPI: tabm
"""

from pathlib import Path  # Path = object-oriented filesystem paths (nicer than raw strings)
from typing import Literal  # Literal["a","b"] = a type hint that allows only those exact values

import numpy as np
import torch
import torch.nn as nn  # nn = neural-network building blocks (layers, Module base class, losses)
from tabm import TabM as TabMModule  # rename on import so our own class can also be called "TabM"

from ports_dfl.config import DEVICE  # the torch device ("cuda" or "cpu") chosen project-wide
from ports_dfl.models.base import BaseModel
from ports_dfl.train.pto import TrainConfig, predict_pto, train_pto


# A leading underscore (``_TabMRegressor``) signals "internal/private" by convention:
# it is a helper used only inside this module, not part of the public API.
# "(nn.Module)" means this class INHERITS from PyTorch's Module, so it can hold
# learnable parameters and be trained by the optimizer.
class _TabMRegressor(nn.Module):
    """Thin wrapper that runs TabM forward and averages over the ensemble dim.

    TabM's forward returns ``(batch, k_ensemble, d_out)``. For point-prediction
    PtO training we average across ``k`` to get a single scalar per row.
    """

    # def __init__(self, ...) -> None: the constructor, run when you create the object.
    # ``self`` is the object being built; ``-> None`` says it returns nothing.
    # Each ``name: type`` is a type hint; ``arch_type: Literal[...] = "tabm"`` restricts
    # the value to one of the listed strings and defaults to "tabm".
    def __init__(
        self,
        input_dim: int,
        k_ensemble: int,
        n_blocks: int,
        d_block: int,
        dropout: float,
        arch_type: Literal["tabm", "tabm-mini", "tabm-packed"] = "tabm",
    ) -> None:
        # super().__init__() runs nn.Module's own constructor first; required so PyTorch
        # can register parameters/submodules. Skipping it breaks training.
        super().__init__()
        # Build the real TabM network and store it as a submodule. ``self.tabm = ...``
        # auto-registers its weights with PyTorch.
        self.tabm = TabMModule(
            n_num_features=input_dim,    # number of numeric input columns
            cat_cardinalities=None,      # no categorical features in this pipeline
            d_out=1,                     # single regression output per ensemble member
            num_embeddings=None,         # no special numeric-feature embeddings
            n_blocks=n_blocks,           # depth (how many MLP blocks are stacked)
            d_block=d_block,             # width of each block (hidden units)
            dropout=dropout,
            k=k_ensemble,                # number of "virtual" ensemble members (BatchEnsemble)
            arch_type=arch_type,
            activation="ReLU",
            start_scaling_init="random-signs",  # init scheme that decorrelates the k members
        )

    # forward() defines what happens when you call the module like a function: out = self(x).
    # PyTorch calls it during both training and inference. ``x: torch.Tensor`` is the input batch.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TabM expects x_num kwarg; output is (batch, k, 1)
        out = self.tabm(x_num=x)
        # Average ensemble members -> (batch, 1)
        # dim=1 is the ensemble axis; mean over it collapses the k members into one prediction.
        return out.mean(dim=1)


# "(BaseModel)" means TabM INHERITS the abstract interface (fit/predict/save/load).
# Because BaseModel marks those with @abstractmethod, Python refuses to create a TabM
# unless all of them are implemented below.
class TabM(BaseModel):
    """BaseModel-compatible TabM wrapper using the project's PtO trainer."""

    # All arguments after ``input_dim`` have defaults (``name: type = value``), so callers
    # can override only the hyperparameters they care about.
    def __init__(
        self,
        input_dim: int,
        k_ensemble: int = 32,
        hidden_dim: int = 384,
        depth: int = 3,
        dropout: float = 0.1,
        arch_type: Literal["tabm", "tabm-mini", "tabm-packed"] = "tabm",
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 16,
        grad_clip: float = 1.0,
        use_amp: bool = True,
    ) -> None:
        # Save every hyperparameter on ``self`` so it can be reused later (e.g. to rebuild
        # the network in load()) and reported back via _init_kwargs().
        self.input_dim = input_dim
        self.k_ensemble = k_ensemble
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout
        self.arch_type = arch_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        # Build the actual PyTorch network now. Note the name mapping into TabM's vocabulary:
        # ``depth`` -> n_blocks, ``hidden_dim`` -> d_block.
        self.module = _TabMRegressor(
            input_dim=input_dim,
            k_ensemble=k_ensemble,
            n_blocks=depth,
            d_block=hidden_dim,
            dropout=dropout,
            arch_type=arch_type,
        )
        # Trailing-underscore attribute (scikit-learn convention) = "set only after fitting".
        # Holds the TrainResult once fit() runs; None means "not trained yet".
        self.train_result_ = None

    # Bundle this object's training hyperparameters into the TrainConfig the PtO loop expects.
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

    # ``X_val: np.ndarray | None = None`` — the "| None" means the arg may be an array OR None
    # (i.e. optional). The return type ``"TabM"`` is a string because the class isn't fully
    # defined yet at this point; it means "returns the same object type, for method chaining".
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TabM":
        # No validation split supplied -> carve one out for early stopping.
        if X_val is None or y_val is None:
            # Hold out 10% of rows (at least 1). int() truncates toward zero.
            n_val = max(int(len(X_train) * 0.1), 1)
            # Negative slicing: [-n_val:] = last n_val rows (val); [:-n_val] = everything before (train).
            # NOTE: assumes the rows are already shuffled upstream; this takes the *tail* as val.
            X_val, y_val = X_train[-n_val:], y_train[-n_val:]
            X_train, y_train = X_train[:-n_val], y_train[:-n_val]
        # Hand off to the shared PtO loop; it trains, early-stops, and restores best weights.
        self.train_result_ = train_pto(
            self.module, X_train, y_train, X_val, y_val, self._train_config()
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Use a bigger batch at inference (no gradients/backprop to store) -> faster.
        return predict_pto(self.module, X, batch_size=self.batch_size * 4)

    # ``path: Path | str`` accepts either a Path object or a plain string.
    def save(self, path: Path | str) -> None:
        path = Path(path)  # normalize to a Path so the methods below always work
        # Create the parent folder if missing; parents=True makes intermediate dirs,
        # exist_ok=True avoids an error if it already exists.
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save a dict with both the hyperparameters (to rebuild the architecture) and the
        # learned weights (state_dict). Storing both lets load() reconstruct everything.
        torch.save(
            {
                "init_kwargs": self._init_kwargs(),
                "state_dict": self.module.state_dict(),
            },
            path,
        )

    def load(self, path: Path | str) -> "TabM":
        # weights_only=False allows loading the Python dict (not just raw tensors).
        # Only safe for checkpoints you trust, since it can execute arbitrary pickled code.
        # map_location=DEVICE puts tensors on the right device regardless of where they were saved.
        ckpt = torch.load(Path(path), map_location=DEVICE, weights_only=False)
        # Restore each saved hyperparameter onto self. ``.items()`` yields (key, value) pairs;
        # setattr(self, k, v) is the dynamic form of ``self.<k> = v``.
        for k, v in ckpt["init_kwargs"].items():
            setattr(self, k, v)
        # Rebuild the empty network with the restored hyperparameters, then pour the weights in.
        self.module = _TabMRegressor(
            input_dim=self.input_dim,
            k_ensemble=self.k_ensemble,
            n_blocks=self.depth,
            d_block=self.hidden_dim,
            dropout=self.dropout,
            arch_type=self.arch_type,
        )
        self.module.load_state_dict(ckpt["state_dict"])  # copy learned weights into the new module
        self.module.to(DEVICE)  # move parameters onto the active device (CPU/GPU)
        return self

    # Snapshot of every constructor argument, used by save() so load() can rebuild an identical model.
    def _init_kwargs(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "k_ensemble": self.k_ensemble,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "dropout": self.dropout,
            "arch_type": self.arch_type,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "grad_clip": self.grad_clip,
            "use_amp": self.use_amp,
        }

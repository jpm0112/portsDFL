"""Linear (Ridge) regressor for service-time prediction.

Tier 1 of the model lineup. A single ``nn.Linear`` head trained under MSE
with optional L2 (via AdamW ``weight_decay``). Acts as the foundational
DFL backbone, matching standard practice in the SPO+ literature.
"""

# Path = an object for file paths that works on any OS (better than raw strings).
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn  # the neural-network building blocks (layers, etc.)

from ports_dfl.config import DEVICE
from ports_dfl.models.base import BaseModel
from ports_dfl.train.pto import TrainConfig, predict_pto, train_pto


# `class _LinearHead(nn.Module)` = inherit from torch's base layer class.
# The leading underscore in `_LinearHead` is a convention meaning "private /
# internal" — not meant to be imported/used outside this file.
class _LinearHead(nn.Module):
    """A single fully connected layer mapping features -> scalar prediction."""

    # `def __init__(self, ...) -> None:` = the constructor (runs when you create
    # the object). `self` = the object being built. `input_dim: int` is a type
    # hint (the arg should be an int). `-> None` = this function returns nothing.
    def __init__(self, input_dim: int) -> None:
        # super().__init__() runs nn.Module's own setup; required before adding layers.
        super().__init__()
        # nn.Linear(in, out) = a layer computing y = x @ W + b.
        # Here out=1, so it maps `input_dim` features to a single number.
        self.fc = nn.Linear(input_dim, 1)

    # forward() defines what the layer computes on input `x`. PyTorch calls this
    # automatically when you do `module(x)`. `x` is a tensor (an array on CPU/GPU).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# Inherits from BaseModel (the abstract interface), so it must provide
# fit/predict/save/load. Tuning and evaluation code can then treat every model
# the same way regardless of what's inside.
class LinearRegressor(BaseModel):
    """Ridge linear regression in PyTorch (BaseModel-compatible).

    Hyperparameters that affect training (lr, weight_decay, batch_size,
    max_epochs, patience, grad_clip) are accepted via constructor kwargs
    and routed into a ``TrainConfig`` at fit time.

    Example:
        >>> model = LinearRegressor(input_dim=42, lr=1e-3, weight_decay=1e-3)
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> preds = model.predict(X_val)
    """

    def __init__(
        self,
        input_dim: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 20,
        grad_clip: float = 1.0,
    ) -> None:
        # Store each hyperparameter on `self` so other methods can read them later.
        self.input_dim = input_dim
        self.lr = lr  # learning rate: how big each gradient step is
        self.weight_decay = weight_decay  # L2 penalty (the "Ridge" part)
        self.batch_size = batch_size
        self.max_epochs = max_epochs  # upper limit on passes over the data
        self.patience = patience  # early-stopping: epochs to wait without improvement
        self.grad_clip = grad_clip  # cap gradient size to avoid exploding updates
        # Build the actual trainable layer now (weights start random/untrained).
        self.module = _LinearHead(input_dim)
        # Trailing underscore (`train_result_`) is a scikit-learn convention for
        # "this attribute only exists after fit() has run". None means "not fitted yet".
        self.train_result_ = None  # populated by fit()

    # Helper that packs this model's hyperparameters into the TrainConfig object
    # the shared training loop expects. Leading underscore = internal helper.
    def _train_config(self) -> TrainConfig:
        """Build a TrainConfig from the model's hyperparameters."""
        return TrainConfig(
            lr=self.lr,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            patience=self.patience,
            grad_clip=self.grad_clip,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LinearRegressor":
        """Train the linear head with early stopping on val MAE.

        If no validation set is supplied, a 10% holdout is carved out from
        ``X_train`` deterministically (last 10% of rows). For real CV runs
        the caller should always provide an explicit validation fold.
        """
        # If the caller gave no validation data, split off the last 10% of rows.
        if X_val is None or y_val is None:
            # int() truncates toward zero; max(..., 1) guarantees at least 1 val row.
            n_val = max(int(len(X_train) * 0.1), 1)
            # `arr[-n:]` = the last n rows; `arr[:-n]` = everything except the last n.
            X_val, y_val = X_train[-n_val:], y_train[-n_val:]
            X_train, y_train = X_train[:-n_val], y_train[:-n_val]

        # Hand everything to the shared training loop; it returns a result object
        # (loss history, best epoch, etc.) which we stash for later inspection.
        self.train_result_ = train_pto(
            self.module, X_train, y_train, X_val, y_val, self._train_config()
        )
        return self  # return self so calls can chain, e.g. model.fit(...).predict(...)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Bigger batch at inference (×4) is fine: no gradients are stored, so we
        # can afford more rows per batch for speed.
        return predict_pto(self.module, X, batch_size=self.batch_size * 4)

    # `path: Path | str` = the arg may be either a Path or a str (the `|` means "or").
    def save(self, path: Path | str) -> None:
        path = Path(path)  # normalize a possible str into a Path object
        # Create the parent folder if missing. parents=True makes intermediate
        # folders too; exist_ok=True avoids an error if it already exists.
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save a plain dict holding the hyperparameters AND the learned weights.
        # state_dict() is torch's name for "all the layer's tensors" (W and b).
        torch.save(
            {
                "input_dim": self.input_dim,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "grad_clip": self.grad_clip,
                "state_dict": self.module.state_dict(),
            },
            path,
        )

    def load(self, path: Path | str) -> "LinearRegressor":
        # torch.load reads the dict we saved above. map_location=DEVICE puts the
        # tensors on the right CPU/GPU; weights_only=True is the safe mode that
        # refuses to execute arbitrary pickled code from the file.
        ckpt = torch.load(Path(path), map_location=DEVICE, weights_only=True)
        # Re-cast each value to its expected type (loading can yield numpy scalars, etc.).
        self.input_dim = int(ckpt["input_dim"])
        self.lr = float(ckpt["lr"])
        self.weight_decay = float(ckpt["weight_decay"])
        self.batch_size = int(ckpt["batch_size"])
        self.max_epochs = int(ckpt["max_epochs"])
        self.patience = int(ckpt["patience"])
        self.grad_clip = float(ckpt["grad_clip"])
        # Build a fresh layer of the right shape, then pour the saved weights into it.
        self.module = _LinearHead(self.input_dim)
        self.module.load_state_dict(ckpt["state_dict"])
        self.module.to(DEVICE)  # move the weights onto the active device (CPU/GPU)
        return self

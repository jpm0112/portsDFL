"""Linear (Ridge) regressor for service-time prediction.

Tier 1 of the model lineup. A single ``nn.Linear`` head trained under MSE
with optional L2 (via AdamW ``weight_decay``). Acts as the foundational
DFL backbone, matching standard practice in the SPO+ literature.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ports_dfl.config import DEVICE
from ports_dfl.models.base import BaseModel
from ports_dfl.train.pto import TrainConfig, predict_pto, train_pto


class _LinearHead(nn.Module):
    """A single fully connected layer mapping features -> scalar prediction."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


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
        self.input_dim = input_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.module = _LinearHead(input_dim)
        self.train_result_ = None  # populated by fit()

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
        if X_val is None or y_val is None:
            n_val = max(int(len(X_train) * 0.1), 1)
            X_val, y_val = X_train[-n_val:], y_train[-n_val:]
            X_train, y_train = X_train[:-n_val], y_train[:-n_val]

        self.train_result_ = train_pto(
            self.module, X_train, y_train, X_val, y_val, self._train_config()
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return predict_pto(self.module, X, batch_size=self.batch_size * 4)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
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
        ckpt = torch.load(Path(path), map_location=DEVICE, weights_only=True)
        self.input_dim = int(ckpt["input_dim"])
        self.lr = float(ckpt["lr"])
        self.weight_decay = float(ckpt["weight_decay"])
        self.batch_size = int(ckpt["batch_size"])
        self.max_epochs = int(ckpt["max_epochs"])
        self.patience = int(ckpt["patience"])
        self.grad_clip = float(ckpt["grad_clip"])
        self.module = _LinearHead(self.input_dim)
        self.module.load_state_dict(ckpt["state_dict"])
        self.module.to(DEVICE)
        return self

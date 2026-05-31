"""TabM (parameter-efficient ensemble of MLPs) wrapper.

Tier 3 of the model lineup. TabM (Gorishniy et al., ICLR 2025) achieves
ensemble-level performance with a single shared backbone via the BatchEnsemble
trick. Top-ranked on TALENT/TabArena 2025 benchmarks.

Reference: https://github.com/yandex-research/tabm
PyPI: tabm
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from tabm import TabM as TabMModule  # rename so our own class can also be called "TabM"

from ports_dfl.config import DEVICE
from ports_dfl.models.base import BaseModel
from ports_dfl.train.pto import TrainConfig, predict_pto, train_pto


class _TabMRegressor(nn.Module):
    """Thin wrapper that runs TabM forward and averages over the ensemble dim.

    TabM's forward returns ``(batch, k_ensemble, d_out)``. For point-prediction
    PtO training we average across ``k`` to get a single scalar per row.
    """

    def __init__(
        self,
        input_dim: int,
        k_ensemble: int,
        n_blocks: int,
        d_block: int,
        dropout: float,
        arch_type: Literal["tabm", "tabm-mini", "tabm-packed"] = "tabm",
    ) -> None:
        super().__init__()
        self.tabm = TabMModule(
            n_num_features=input_dim,
            cat_cardinalities=None,      # no categorical features in this pipeline
            d_out=1,                     # single regression output per ensemble member
            num_embeddings=None,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
            k=k_ensemble,                # number of "virtual" ensemble members (BatchEnsemble)
            arch_type=arch_type,
            activation="ReLU",
            start_scaling_init="random-signs",  # init scheme that decorrelates the k members
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TabM expects x_num kwarg; output is (batch, k, 1)
        out = self.tabm(x_num=x)
        # Average over the ensemble axis -> (batch, 1)
        return out.mean(dim=1)


class TabM(BaseModel):
    """BaseModel-compatible TabM wrapper using the project's PtO trainer."""

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
        # Name mapping into TabM's vocabulary: depth -> n_blocks, hidden_dim -> d_block.
        self.module = _TabMRegressor(
            input_dim=input_dim,
            k_ensemble=k_ensemble,
            n_blocks=depth,
            d_block=hidden_dim,
            dropout=dropout,
            arch_type=arch_type,
        )
        self.train_result_ = None

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

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TabM":
        if X_val is None or y_val is None:
            n_val = max(int(len(X_train) * 0.1), 1)
            # NOTE: assumes rows are already shuffled upstream; this takes the *tail* as val.
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
            {
                "init_kwargs": self._init_kwargs(),
                "state_dict": self.module.state_dict(),
            },
            path,
        )

    def load(self, path: Path | str) -> "TabM":
        # weights_only=False allows loading the stored dict (not just raw tensors);
        # only safe for checkpoints you trust, since it can execute arbitrary pickled code.
        ckpt = torch.load(Path(path), map_location=DEVICE, weights_only=False)
        for k, v in ckpt["init_kwargs"].items():
            setattr(self, k, v)
        self.module = _TabMRegressor(
            input_dim=self.input_dim,
            k_ensemble=self.k_ensemble,
            n_blocks=self.depth,
            d_block=self.hidden_dim,
            dropout=self.dropout,
            arch_type=self.arch_type,
        )
        self.module.load_state_dict(ckpt["state_dict"])
        self.module.to(DEVICE)
        return self

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

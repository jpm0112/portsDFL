"""Generic predict-then-optimize (PtO) training loop.

Used by linear, RealMLP, TabM, and NODE under MSE loss. Provides:
  - device-aware tensor placement (CUDA when available)
  - ``pin_memory`` + multi-worker DataLoader for GPU throughput
  - cosine LR schedule
  - early stopping on val MAE
  - optional mixed precision (``torch.cuda.amp``)
  - gradient clipping
"""

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ports_dfl.config import DEVICE, SEED, set_seed


@dataclass
class TrainConfig:
    """Hyperparameters for the PtO training loop."""

    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20                  # early-stop patience on val MAE
    grad_clip: float = 1.0
    use_amp: bool = False               # mixed precision; default off
    num_workers: int = 0                # set to 2-4 if GPU is the bottleneck
    pin_memory: bool = True
    cosine_eta_min: float = 1e-6
    seed: int = SEED
    extra: dict = field(default_factory=dict)


@dataclass
class TrainResult:
    """Per-epoch trace and final metrics from a training run."""

    train_loss_history: list[float]
    val_mae_history: list[float]
    best_epoch: int
    best_val_mae: float
    epochs_run: int


def _build_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Wrap numpy arrays into a TensorDataset + DataLoader on CPU.

    Tensors are kept on CPU and moved to device per batch — saves GPU memory
    and lets ``pin_memory`` work.
    """
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )


def _to_device(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
    """Move tensors to the configured device, non-blocking when possible."""
    return (t.to(DEVICE, non_blocking=True) for t in tensors)


def train_pto(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Train a PyTorch regression module under MSE loss with early stopping.

    Args:
        model: Module returning shape ``(batch, 1)`` predictions.
        X_train: Preprocessed training features.
        y_train: Training targets.
        X_val: Preprocessed validation features.
        y_val: Validation targets.
        config: Optional ``TrainConfig`` override (uses defaults otherwise).

    Returns:
        A :class:`TrainResult` with loss/MAE histories and best-epoch info.
        The model is restored to its best-MAE state in place.
    """
    cfg = config or TrainConfig()
    set_seed(cfg.seed)
    model = model.to(DEVICE)

    train_loader = _build_loader(
        X_train, y_train, cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )
    val_loader = _build_loader(
        X_val, y_val, cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs, eta_min=cfg.cosine_eta_min,
    )
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and torch.cuda.is_available())

    train_loss_history: list[float] = []
    val_mae_history: list[float] = []
    best_val_mae = float("inf")
    best_state: dict = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(cfg.max_epochs):
        # --- Train ---------------------------------------------------------
        model.train()
        running_loss = 0.0
        n_seen = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = _to_device(X_batch, y_batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                pred = model(X_batch)
                loss = loss_fn(pred, y_batch)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * X_batch.size(0)
            n_seen += X_batch.size(0)
        scheduler.step()
        train_loss_history.append(running_loss / max(n_seen, 1))

        # --- Validate ------------------------------------------------------
        model.eval()
        abs_err_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = _to_device(X_batch, y_batch)
                pred = model(X_batch)
                abs_err_sum += (pred - y_batch).abs().sum().item()
                n_val += X_batch.size(0)
        val_mae = abs_err_sum / max(n_val, 1)
        val_mae_history.append(val_mae)

        # --- Early stopping -----------------------------------------------
        if val_mae < best_val_mae - 1e-6:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    model.load_state_dict(best_state)

    return TrainResult(
        train_loss_history=train_loss_history,
        val_mae_history=val_mae_history,
        best_epoch=best_epoch,
        best_val_mae=best_val_mae,
        epochs_run=len(train_loss_history),
    )


@torch.no_grad()
def predict_pto(model: nn.Module, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Run inference and return a 1D numpy array."""
    model = model.to(DEVICE).eval()
    X_t = torch.as_tensor(X, dtype=torch.float32)
    outputs: list[np.ndarray] = []
    for i in range(0, X_t.size(0), batch_size):
        batch = X_t[i : i + batch_size].to(DEVICE, non_blocking=True)
        outputs.append(model(batch).detach().cpu().numpy().ravel())
    return np.concatenate(outputs, axis=0)

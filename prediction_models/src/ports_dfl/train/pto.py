"""Generic predict-then-optimize (PtO) training loop.

Used by linear, RealMLP, TabM, and NODE under MSE loss. Provides:
  - device-aware tensor placement (CUDA when available)
  - ``pin_memory`` + multi-worker DataLoader for GPU throughput
  - cosine LR schedule
  - early stopping on val MAE
  - optional mixed precision (``torch.cuda.amp``)
  - gradient clipping
"""

# `dataclass` (used below as @dataclass) auto-generates __init__/__repr__ for us.
# `field` lets us give a dataclass attribute a more complex default (see `extra`).
from dataclasses import dataclass, field
# Type hints only: `Iterable` describes "something you can loop over". Hints
# document intent and help editors/type-checkers; Python does not enforce them.
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ports_dfl.config import DEVICE, SEED, set_seed


# `@dataclass` is a decorator: it modifies the class right below it. Here it
# turns this class into a simple config "struct" where each line `name: type = default`
# becomes a constructor argument with that default value.
@dataclass
class TrainConfig:
    """Hyperparameters for the PtO training loop."""

    # `lr: float = 1e-3` reads as: attribute `lr`, hinted as a float, default 0.001.
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
    # Mutable defaults (dict/list) must NOT be written as `extra: dict = {}` — every
    # instance would share ONE dict. `field(default_factory=dict)` builds a fresh
    # empty dict per instance instead. This is the correct, safe pattern.
    extra: dict = field(default_factory=dict)


@dataclass
class TrainResult:
    """Per-epoch trace and final metrics from a training run."""

    # `list[float]` is a type hint meaning "a list whose items are floats".
    # These have no `= default`, so they are REQUIRED constructor arguments.
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
    # `torch.as_tensor` wraps the numpy array as a tensor without copying when it can.
    # We force float32 because models expect 32-bit floats by default.
    X_t = torch.as_tensor(X, dtype=torch.float32)
    # `.reshape(-1, 1)` turns a flat targets vector of length N into shape (N, 1).
    # The `-1` means "infer this dimension"; the result is one column. This matches
    # the model output shape (batch, 1) so the loss lines up element-wise.
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    # Pair features+targets so the DataLoader yields (X_batch, y_batch) together.
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # pin_memory only helps (and is only valid) when a CUDA GPU exists; guard it.
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
    )


# `*tensors` collects all positional arguments into one tuple, so you can pass
# any number of tensors. The hint `Iterable[...]` says the return is loopable.
def _to_device(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
    """Move tensors to the configured device, non-blocking when possible."""
    # This is a GENERATOR EXPRESSION (parentheses, not brackets): it produces each
    # moved tensor lazily as you iterate. `non_blocking=True` lets the CPU->GPU copy
    # overlap with compute when memory is pinned. Unpacking `a, b = _to_device(a, b)`
    # drains the generator in order, which is why callers can unpack it directly.
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
    # `config or TrainConfig()` returns `config` if it is truthy (not None),
    # otherwise falls back to a fresh default config. This is the common Python
    # idiom for "use the argument, or a default if none was given".
    cfg = config or TrainConfig()
    set_seed(cfg.seed)              # make this run reproducible (same seed -> same result)
    model = model.to(DEVICE)        # put model weights on GPU/CPU as configured

    train_loader = _build_loader(
        X_train, y_train, cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )
    val_loader = _build_loader(
        X_val, y_val, cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )

    # AdamW = Adam optimizer with proper weight decay (L2-style regularization).
    # `model.parameters()` hands the optimizer every weight tensor it should update.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    # Cosine schedule: smoothly lowers the learning rate from `lr` toward `eta_min`
    # over `T_max` epochs (one .step() per epoch). Helps fine convergence late on.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs, eta_min=cfg.cosine_eta_min,
    )
    loss_fn = nn.MSELoss()          # mean squared error: average of (pred - target)^2
    # GradScaler supports mixed-precision (AMP) training; only enabled with AMP + CUDA.
    # When disabled it is a no-op, so the same code path works on CPU.
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and torch.cuda.is_available())

    train_loss_history: list[float] = []
    val_mae_history: list[float] = []
    best_val_mae = float("inf")     # start "worst possible" so the first epoch always beats it
    # Snapshot the model's current weights as the initial "best" so we can restore
    # them later. `.state_dict()` is the dict of all parameters/buffers.
    # `.detach()` removes gradient tracking, `.cpu()` moves to CPU, `.clone()` makes
    # an independent copy so future training updates do not overwrite this snapshot.
    # This is a DICT COMPREHENSION: {key: value for key, value in iterable}.
    best_state: dict = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    best_epoch = 0
    epochs_no_improve = 0           # counts consecutive epochs without improvement

    for epoch in range(cfg.max_epochs):
        # --- Train ---------------------------------------------------------
        model.train()               # put model in TRAIN mode (enables dropout/batchnorm updates)
        running_loss = 0.0          # sum of (batch loss * batch size), to average later
        n_seen = 0                  # total samples processed this epoch
        for X_batch, y_batch in train_loader:
            # Unpacking the generator returned by _to_device moves both to the device.
            X_batch, y_batch = _to_device(X_batch, y_batch)
            # Clear old gradients before computing new ones. `set_to_none=True` frees
            # the gradient tensors (slightly faster / less memory) instead of zeroing them.
            optimizer.zero_grad(set_to_none=True)
            # `with` runs this block under autocast: ops use float16 where safe (AMP).
            # When AMP is off, autocast is effectively a no-op.
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                pred = model(X_batch)               # forward pass -> shape (batch, 1)
                loss = loss_fn(pred, y_batch)        # scalar MSE for this batch
            # `scaler.scale(loss)` scales the loss up before backprop to avoid float16
            # underflow; `.backward()` computes gradients of all parameters.
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                # Must unscale gradients back to true values BEFORE clipping by norm,
                # otherwise the clip threshold would be applied to scaled gradients.
                scaler.unscale_(optimizer)
                # Cap the global gradient norm to `grad_clip` to prevent exploding updates.
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)  # apply the optimizer step (skips it if grads are inf/nan)
            scaler.update()         # adjust the scale factor for next iteration
            # `.item()` pulls the scalar loss out as a plain Python float. Multiply by
            # batch size so the running sum is correctly weighted (last batch may be smaller).
            running_loss += loss.item() * X_batch.size(0)
            n_seen += X_batch.size(0)
        scheduler.step()            # advance the cosine LR schedule once per epoch
        # `max(n_seen, 1)` guards against divide-by-zero if a loader were empty.
        train_loss_history.append(running_loss / max(n_seen, 1))

        # --- Validate ------------------------------------------------------
        model.eval()                # EVAL mode: disables dropout, uses running batchnorm stats
        abs_err_sum = 0.0           # sum of absolute errors over the whole val set
        n_val = 0
        # `torch.no_grad()` turns off gradient tracking inside the block: faster and
        # uses less memory since we are only measuring, not training.
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = _to_device(X_batch, y_batch)
                pred = model(X_batch)
                # MAE numerator: sum |pred - target| across this batch.
                abs_err_sum += (pred - y_batch).abs().sum().item()
                n_val += X_batch.size(0)
        val_mae = abs_err_sum / max(n_val, 1)   # mean absolute error (guarded divide)
        val_mae_history.append(val_mae)

        # --- Early stopping -----------------------------------------------
        # Improvement must beat the best by at least 1e-6 to count (ignores noise).
        if val_mae < best_val_mae - 1e-6:
            best_val_mae = val_mae
            best_epoch = epoch
            # Re-snapshot the (now better) weights so we can restore them at the end.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0   # reset the patience counter on every improvement
        else:
            epochs_no_improve += 1
            # Stop early if val MAE has not improved for `patience` epochs in a row.
            if epochs_no_improve >= cfg.patience:
                break

    # Restore the best-MAE weights into the model (in place), so the returned model
    # is the best one seen, not whatever the last epoch produced.
    model.load_state_dict(best_state)

    return TrainResult(
        train_loss_history=train_loss_history,
        val_mae_history=val_mae_history,
        best_epoch=best_epoch,
        best_val_mae=best_val_mae,
        epochs_run=len(train_loss_history),
    )


# `@torch.no_grad()` as a decorator wraps the whole function so NO call inside it
# tracks gradients — exactly what you want for inference (faster, less memory).
@torch.no_grad()
def predict_pto(model: nn.Module, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Run inference and return a 1D numpy array."""
    model = model.to(DEVICE).eval()                 # ensure correct device + eval mode
    X_t = torch.as_tensor(X, dtype=torch.float32)
    # FIX: handle empty input. Without this, the loop runs zero times and
    # `np.concatenate([])` raises "need at least one array to concatenate".
    if X_t.size(0) == 0:
        return np.empty((0,), dtype=np.float32)
    outputs: list[np.ndarray] = []
    # Step through rows in chunks of `batch_size` so a huge X does not blow up memory.
    for i in range(0, X_t.size(0), batch_size):
        # `X_t[i : i + batch_size]` slices rows i .. i+batch_size (a Python slice).
        batch = X_t[i : i + batch_size].to(DEVICE, non_blocking=True)
        # `.detach()` drops grad info, `.cpu()` brings it back from GPU, `.numpy()`
        # converts to numpy, `.ravel()` flattens (batch, 1) -> (batch,).
        outputs.append(model(batch).detach().cpu().numpy().ravel())
    # Glue all the per-batch 1D arrays into one long 1D array.
    return np.concatenate(outputs, axis=0)

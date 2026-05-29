"""DFL training via PyEPO's blackbox differentiation for the discrete BAP.

Used when predicted parameters enter constraints (not just the linear
objective). The DBAP from ``optim/discrete_bap.py`` is exactly this case:
predicted service times τ̂ shift big-M precedence inequalities, so SPO+
no longer applies.

Pipeline per gradient step:
    1.  Predict τ̂ for the N vessels in the instance.
    2.  ``blackboxOpt(τ̂)`` solves the DBAP under τ̂, returning start times s*.
    3.  Compute true cost  L = Σᵢ wᵢ (s*[i] + τ_trueᵢ)   ← the regret-style loss.
    4.  Backpropagate through ``blackboxOpt`` (Pogančić et al., 2020 STE).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from pyepo.func import blackboxOpt
from torch.utils.data import DataLoader, TensorDataset

from ports_dfl.config import DEVICE, SEED, set_seed
from ports_dfl.optim.discrete_bap import (
    BAPInstance,
    DiscreteBAP,
    extract_decision,
    schedule_cost_under_true_tau,
)


@dataclass
class DFLBlackboxConfig:
    """Hyperparameters for the blackbox-DFL training loop."""

    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 4           # number of instances per gradient step
    max_epochs: int = 20
    patience: int = 6
    grad_clip: float = 1.0
    blackbox_lambd: float = 10.0  # interpolation strength (Pogancic 2020)
    processes: int = 1            # parallel solver workers
    seed: int = SEED              # reseed RNGs at training entry


@dataclass
class DFLBlackboxResult:
    """Trace of a blackbox-DFL run."""

    train_loss_history: list[float]
    val_regret_history: list[float]
    best_epoch: int
    best_val_regret: float
    epochs_run: int


def _evaluate_regret(
    model: nn.Module,
    X_inst: np.ndarray,
    tau_inst: np.ndarray,
    instance: BAPInstance,
    optmodel: DiscreteBAP,
) -> float:
    """Mean regret across val instances (true cost units, weight·hours).

    The decision (x, z) is solved under the predictor's τ̂, then re-evaluated
    feasibly under true τ. The full-information (FI) decision is solved
    under true τ — that's the post-hoc optimal benchmark in DFL terminology.
    """
    model.eval()
    regrets = []
    with torch.no_grad():
        for x_inst, tau_true in zip(X_inst, tau_inst, strict=True):
            x_t = torch.as_tensor(x_inst, dtype=torch.float32, device=DEVICE)
            tau_pred = model(x_t).detach().cpu().numpy().ravel()

            optmodel.setObj(tau_pred)
            optmodel.solve()
            assign_pred, order_pred = extract_decision(optmodel)
            cost_pred, _ = schedule_cost_under_true_tau(
                assign_pred, order_pred, tau_true, instance.arrivals, instance.weights
            )

            optmodel.setObj(tau_true)
            optmodel.solve()
            assign_fi, order_fi = extract_decision(optmodel)
            cost_fi, _ = schedule_cost_under_true_tau(
                assign_fi, order_fi, tau_true, instance.arrivals, instance.weights
            )

            regrets.append(cost_pred - cost_fi)
    return float(np.mean(regrets))


def train_dfl_blackbox(
    model: nn.Module,
    X_inst_train: np.ndarray,
    tau_inst_train: np.ndarray,
    X_inst_val: np.ndarray,
    tau_inst_val: np.ndarray,
    instance: BAPInstance,
    config: DFLBlackboxConfig | None = None,
) -> DFLBlackboxResult:
    """Train a regression module against a real DBAP via blackbox DFL.

    The "loss" is the true weighted completion time of the schedule
    produced under predicted τ. Gradients flow back through PyEPO's
    blackbox interpolation.

    Args:
        model:           PyTorch module mapping (n_vessels × n_features) → (n_vessels,).
        X_inst_train:    training instances, shape (n_inst, n_vessels, n_features).
        tau_inst_train:  ground-truth service times, shape (n_inst, n_vessels).
        X_inst_val:      validation instances.
        tau_inst_val:    validation ground-truth τ.
        instance:        BAPInstance specifying N, M, arrivals, weights.
        config:          Optional DFLBlackboxConfig.

    Returns:
        DFLBlackboxResult with per-epoch loss/regret traces.
    """
    cfg = config or DFLBlackboxConfig()
    set_seed(cfg.seed)
    model = model.to(DEVICE)
    optmodel = DiscreteBAP(instance)
    bb_opt = blackboxOpt(
        optmodel, lambd=cfg.blackbox_lambd, processes=cfg.processes
    )

    weights_t = torch.as_tensor(instance.weights, dtype=torch.float32, device=DEVICE)

    train_ds = TensorDataset(
        torch.as_tensor(X_inst_train, dtype=torch.float32),
        torch.as_tensor(tau_inst_train, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    train_loss_history: list[float] = []
    val_regret_history: list[float] = []
    # Initialize best_state to the warm-start so a never-improves run still
    # restores a valid checkpoint (rather than leaving best_state=None).
    best_state: dict = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    best_val_regret = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for X_b, tau_b in train_loader:
            B, N, _ = X_b.shape
            X_b = X_b.to(DEVICE)
            tau_b = tau_b.to(DEVICE)

            tau_pred = model(X_b.reshape(B * N, -1)).view(B, N)
            optimizer.zero_grad(set_to_none=True)
            # Solve DBAP under predicted τ. Returns shape (B, N) of start times.
            starts = bb_opt(tau_pred)
            # Loss = weighted completion under TRUE τ
            loss = (weights_t * (starts + tau_b)).sum(dim=1).mean()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            running_loss += loss.item() * B
            n_seen += B
        train_loss_history.append(running_loss / max(n_seen, 1))

        val_regret = _evaluate_regret(
            model, X_inst_val, tau_inst_val, instance, optmodel
        )
        val_regret_history.append(val_regret)
        print(
            f"  epoch {epoch + 1:3d}: train_loss={train_loss_history[-1]:.2f} "
            f"val_regret={val_regret:.2f}"
        )

        if val_regret < best_val_regret - 1e-6:
            best_val_regret = val_regret
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    model.load_state_dict(best_state)

    return DFLBlackboxResult(
        train_loss_history=train_loss_history,
        val_regret_history=val_regret_history,
        best_epoch=best_epoch,
        best_val_regret=best_val_regret,
        epochs_run=len(train_loss_history),
    )

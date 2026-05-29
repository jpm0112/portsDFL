"""DFL training via PyEPO's Gaussian-perturbation differentiation
(Berthet et al., NeurIPS 2020, "Learning with Differentiable Perturbed
Optimizers").

Same regret-style loss as :mod:`ports_dfl.train.dfl_blackbox`; the only
thing that changes is the differentiable optimizer wrapper. Instead of
the two-solve finite-difference surrogate of Pogan\\v{c}i\\'c 2020,
Berthet samples ``n_samples`` Gaussian perturbations of the cost
vector, solves the MILP at each, and treats the (smoothed) argmin as a
differentiable function via the Fenchel-Young framework.

Each training step costs roughly ``n_samples + 1`` MILP solves
(``n_samples`` for the forward smoothing, plus one for regret
evaluation on the val set). This is more expensive than blackbox but
generally produces lower-variance gradients.

Pipeline per gradient step:
    1.  Predict τ̂ for the N vessels in the instance.
    2.  ``perturbedOpt(τ̂)`` samples K perturbations, solves
        ``DBAP(τ̂ + σ·ε_k)`` for each, returns the average of the
        resulting start times s*.
    3.  Compute true cost L = Σᵢ wᵢ (s*[i] + τ_trueᵢ).
    4.  Backpropagate through the smoothed argmin (analytic gradient).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from pyepo.func import perturbedOpt
from torch.utils.data import DataLoader, TensorDataset

from ports_dfl.config import DEVICE, SEED, set_seed
from ports_dfl.optim.discrete_bap import (
    BAPInstance,
    DiscreteBAP,
    extract_decision,
    schedule_cost_under_true_tau,
)


@dataclass
class DFLPerturbedConfig:
    """Hyperparameters for the perturbed-DFL training loop."""

    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 4
    max_epochs: int = 20
    patience: int = 6
    grad_clip: float = 1.0
    n_samples: int = 10           # Gaussian perturbation samples (Berthet)
    sigma: float = 1.0            # std of the Gaussian perturbation
    processes: int = 1
    seed: int = SEED


@dataclass
class DFLPerturbedResult:
    """Trace of a perturbed-DFL run."""

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

    Same protocol as :func:`ports_dfl.train.dfl_blackbox._evaluate_regret`.
    """
    model.eval()
    regrets: list[float] = []
    with torch.no_grad():
        for x_inst, tau_true in zip(X_inst, tau_inst, strict=True):
            x_t = torch.as_tensor(x_inst, dtype=torch.float32, device=DEVICE)
            tau_pred = model(x_t).detach().cpu().numpy().ravel()

            optmodel.setObj(tau_pred)
            optmodel.solve()
            assign_pred, order_pred = extract_decision(optmodel)
            cost_pred, _ = schedule_cost_under_true_tau(
                assign_pred, order_pred, tau_true,
                instance.arrivals, instance.weights,
            )

            optmodel.setObj(tau_true)
            optmodel.solve()
            assign_fi, order_fi = extract_decision(optmodel)
            cost_fi, _ = schedule_cost_under_true_tau(
                assign_fi, order_fi, tau_true,
                instance.arrivals, instance.weights,
            )

            regrets.append(cost_pred - cost_fi)
    return float(np.mean(regrets))


def train_dfl_perturbed(
    model: nn.Module,
    X_inst_train: np.ndarray,
    tau_inst_train: np.ndarray,
    X_inst_val: np.ndarray,
    tau_inst_val: np.ndarray,
    instance: BAPInstance,
    config: DFLPerturbedConfig | None = None,
) -> DFLPerturbedResult:
    """Train a regression module against the DBAP via Berthet perturbation.

    Args:
        model:           PyTorch module mapping (N × F) → (N,).
        X_inst_train:    training instances, shape (n_inst, n_vessels, n_features).
        tau_inst_train:  ground-truth service times, shape (n_inst, n_vessels).
        X_inst_val:      validation instances.
        tau_inst_val:    validation ground-truth τ.
        instance:        BAPInstance specifying N, M, arrivals, weights.
        config:          Optional :class:`DFLPerturbedConfig`.

    Returns:
        :class:`DFLPerturbedResult` with per-epoch loss/regret traces.
    """
    cfg = config or DFLPerturbedConfig()
    set_seed(cfg.seed)
    model = model.to(DEVICE)
    optmodel = DiscreteBAP(instance)
    p_opt = perturbedOpt(
        optmodel,
        n_samples=cfg.n_samples,
        sigma=cfg.sigma,
        processes=cfg.processes,
        seed=cfg.seed,
    )

    weights_t = torch.as_tensor(
        instance.weights, dtype=torch.float32, device=DEVICE
    )

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
            starts = p_opt(tau_pred)
            loss = (weights_t * (starts + tau_b)).sum(dim=1).mean()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                )
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

    return DFLPerturbedResult(
        train_loss_history=train_loss_history,
        val_regret_history=val_regret_history,
        best_epoch=best_epoch,
        best_val_regret=best_val_regret,
        epochs_run=len(train_loss_history),
    )

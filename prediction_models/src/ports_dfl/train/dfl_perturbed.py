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
# perturbedOpt makes the MILP differentiable via Berthet et al. (2020): averaging
# solutions over Gaussian-perturbed cost vectors yields an analytic (Fenchel-Young)
# gradient — see this module's docstring.
from pyepo.func import perturbedOpt
from torch.utils.data import DataLoader, TensorDataset

from ports_dfl.config import DEVICE, SEED, set_seed
from bap_optim.discrete_bap import (
    BAPInstance,
    DiscreteBAP,
    extract_decision,
    schedule_cost_under_true_tau,
)


@dataclass
class DFLPerturbedConfig:
    """Hyperparameters for the perturbed-DFL training loop."""

    lr: float = 1e-3              # learning rate for AdamW
    weight_decay: float = 0.0     # L2 regularisation strength
    batch_size: int = 4           # instances per gradient step
    max_epochs: int = 20
    patience: int = 6             # early-stop after this many non-improving epochs
    grad_clip: float = 1.0        # clip gradient norm to tame noisy DFL grads
    n_samples: int = 10           # Gaussian perturbation samples (Berthet)
    sigma: float = 1.0            # std of the Gaussian perturbation
    processes: int = 1            # parallel solver workers
    seed: int = SEED              # reseed RNGs at training entry


@dataclass
class DFLPerturbedResult:
    """Trace of a perturbed-DFL run."""

    train_loss_history: list[float]   # mean training loss per epoch
    val_regret_history: list[float]   # validation regret per epoch
    best_epoch: int                   # epoch with the lowest val regret
    best_val_regret: float            # that lowest regret
    epochs_run: int                   # epochs actually run (early stop may cut it)


# Same regret protocol as the blackbox trainer: it does not use the differentiable
# wrapper at all — it solves the plain MILP twice per instance (under τ̂ and under
# true τ) and compares realised costs.
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
        # strict=True guards against a silent length mismatch between features and labels
        for x_inst, tau_true in zip(X_inst, tau_inst, strict=True):
            x_t = torch.as_tensor(x_inst, dtype=torch.float32, device=DEVICE)
            tau_pred = model(x_t).detach().cpu().numpy().ravel()

            # --- Predicted decision: solve under τ̂, evaluate under true τ ---
            optmodel.setObj(tau_pred)
            optmodel.solve()
            assign_pred, order_pred = extract_decision(optmodel)
            # re-derive feasible starts under reality -> the true cost of acting on τ̂
            cost_pred, _ = schedule_cost_under_true_tau(
                assign_pred, order_pred, tau_true, instance.arrivals,
            )

            # --- Full-information optimum: solve AND evaluate under true τ ---
            optmodel.setObj(tau_true)
            optmodel.solve()
            assign_fi, order_fi = extract_decision(optmodel)
            cost_fi, _ = schedule_cost_under_true_tau(
                assign_fi, order_fi, tau_true, instance.arrivals,
            )

            # regret >= 0: extra cost from deciding on τ̂ instead of true τ
            regrets.append(cost_pred - cost_fi)
    if not regrets:
        # np.mean([]) -> nan would silently disable early stopping (best_val_regret
        # stays inf, no epoch ever "improves"). An empty val set is a config error.
        raise ValueError("No validation instances to evaluate regret on.")
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
    # build the MILP once and re-solve under different τ (mutable τ Param)
    optmodel = DiscreteBAP(instance)
    # n_samples = perturbed cost vectors to average (more = smoother but more solves);
    # sigma = perturbation std; seed makes the sampled perturbations reproducible
    p_opt = perturbedOpt(
        optmodel,
        n_samples=cfg.n_samples,
        sigma=cfg.sigma,
        processes=cfg.processes,
        seed=cfg.seed,
    )

    train_ds = TensorDataset(
        torch.as_tensor(X_inst_train, dtype=torch.float32),
        torch.as_tensor(tau_inst_train, dtype=torch.float32),
    )
    # num_workers=0 keeps loading in-process (avoids forking the solver state)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    train_loss_history: list[float] = []
    val_regret_history: list[float] = []
    # warm-start checkpoint so a never-improving run still restores valid weights
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
            # batch shape (B instances, N vessels, F features)
            B, N, _ = X_b.shape
            X_b = X_b.to(DEVICE)
            tau_b = tau_b.to(DEVICE)

            # score all B*N vessels in one call, then reshape to per-instance (B, N)
            tau_pred = model(X_b.reshape(B * N, -1)).view(B, N)
            optimizer.zero_grad(set_to_none=True)
            # differentiable solve -> (B, N) start times averaged over the n_samples
            # Gaussian perturbations of τ̂; gradient flows through this
            starts = p_opt(tau_pred)
            # DFL loss = realised (unweighted) total completion under the TRUE τ,
            # matching the MILP's weight-blind objective. As in the blackbox
            # trainer, `starts` are the optimizer's own (smoothed) start times,
            # not feasibility-re-derived ones.
            loss = (starts + tau_b).sum(dim=1).mean()
            loss.backward()
            if cfg.grad_clip > 0:
                # stabilise these high-variance DFL gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                )
            optimizer.step()

            # size-weight so the epoch mean is correct even if the last batch is smaller
            running_loss += loss.item() * B
            n_seen += B
        train_loss_history.append(running_loss / max(n_seen, 1))

        # held-out decision quality: the metric we actually select on
        val_regret = _evaluate_regret(
            model, X_inst_val, tau_inst_val, instance, optmodel
        )
        val_regret_history.append(val_regret)
        print(
            f"  epoch {epoch + 1:3d}: train_loss={train_loss_history[-1]:.2f} "
            f"val_regret={val_regret:.2f}"
        )

        # require a real improvement of at least 1e-6 so FP noise doesn't count
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

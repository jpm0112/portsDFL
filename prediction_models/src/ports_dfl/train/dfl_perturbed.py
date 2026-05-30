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

# `from __future__ import annotations` stores all type hints as plain text so we
# can write modern hints (`int | None`) and forward-references on older Pythons.
from __future__ import annotations

# `@dataclass` (used below) auto-generates __init__/__repr__/== for data-holding
# classes from their field declarations.
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn  # PyTorch neural-network building blocks
# perturbedOpt wraps our MILP and makes it differentiable via Berthet et al.
# (2020): it averages solutions over many Gaussian-perturbed cost vectors, which
# yields an analytic (Fenchel-Young) gradient — see this module's docstring.
from pyepo.func import perturbedOpt
from torch.utils.data import DataLoader, TensorDataset  # dataset batching helpers

from ports_dfl.config import DEVICE, SEED, set_seed
from bap_optim.discrete_bap import (
    BAPInstance,
    DiscreteBAP,
    extract_decision,
    schedule_cost_under_true_tau,
)


# Build with `DFLPerturbedConfig()` for defaults, or override fields by name,
# e.g. `DFLPerturbedConfig(n_samples=20, sigma=0.5)`.
@dataclass
class DFLPerturbedConfig:
    """Hyperparameters for the perturbed-DFL training loop."""

    # `name: type = default` — type hint plus default value (see dfl_blackbox.py).
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

    # No defaults: the caller supplies all of these when constructing the result.
    train_loss_history: list[float]   # mean training loss per epoch
    val_regret_history: list[float]   # validation regret per epoch
    best_epoch: int                   # epoch with the lowest val regret
    best_val_regret: float            # that lowest regret
    epochs_run: int                   # epochs actually run (early stop may cut it)


# Module-private helper (leading underscore). `-> float` is the return-type hint.
# This is the SAME regret protocol as the blackbox trainer: it does not use the
# differentiable wrapper at all — it just solves the plain MILP twice per
# instance (under τ̂ and under true τ) and compares realised costs.
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
    model.eval()              # eval mode: deterministic forward pass (no dropout)
    regrets: list[float] = []
    with torch.no_grad():     # disable autograd: we only need predictions here
        # `strict=True` errors if features and labels have mismatched lengths.
        for x_inst, tau_true in zip(X_inst, tau_inst, strict=True):
            x_t = torch.as_tensor(x_inst, dtype=torch.float32, device=DEVICE)
            # Predict τ̂, detach from autograd, move to CPU, to numpy, flatten to (N,).
            tau_pred = model(x_t).detach().cpu().numpy().ravel()

            # --- Predicted decision: solve under τ̂, evaluate under true τ ---
            optmodel.setObj(tau_pred)   # set predicted service times on the MILP
            optmodel.solve()
            assign_pred, order_pred = extract_decision(optmodel)
            # Re-derive feasible starts under reality -> the true cost of acting on τ̂.
            cost_pred, _ = schedule_cost_under_true_tau(
                assign_pred, order_pred, tau_true,
                instance.arrivals, instance.weights,
            )

            # --- Full-information optimum: solve AND evaluate under true τ ---
            optmodel.setObj(tau_true)
            optmodel.solve()
            assign_fi, order_fi = extract_decision(optmodel)
            cost_fi, _ = schedule_cost_under_true_tau(
                assign_fi, order_fi, tau_true,
                instance.arrivals, instance.weights,
            )

            # Regret >= 0: extra cost from deciding on τ̂ instead of true τ.
            regrets.append(cost_pred - cost_fi)
    return float(np.mean(regrets))   # average regret over the val set


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
    # Use the supplied config, or fall back to all-default values.
    cfg = config or DFLPerturbedConfig()
    set_seed(cfg.seed)          # reproducible RNG state across python/numpy/torch
    model = model.to(DEVICE)    # move weights to the compute device
    # Build the MILP once and re-solve it under different τ (mutable τ Param).
    optmodel = DiscreteBAP(instance)
    # Wrap it as a differentiable perturbed optimizer:
    #   n_samples = how many Gaussian-perturbed cost vectors to average over
    #               (more = smoother/lower-variance gradient, but more solves),
    #   sigma     = std of those perturbations,
    #   seed      = makes the sampled perturbations reproducible.
    p_opt = perturbedOpt(
        optmodel,
        n_samples=cfg.n_samples,
        sigma=cfg.sigma,
        processes=cfg.processes,
        seed=cfg.seed,
    )

    # Constant priority weights as a device tensor, reused in the loss each step.
    weights_t = torch.as_tensor(
        instance.weights, dtype=torch.float32, device=DEVICE
    )

    # Pair features with true τ as aligned rows; DataLoader yields shuffled batches.
    train_ds = TensorDataset(
        torch.as_tensor(X_inst_train, dtype=torch.float32),
        torch.as_tensor(tau_inst_train, dtype=torch.float32),
    )
    # num_workers=0 keeps loading in-process (avoids forking the solver state).
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )

    # AdamW optimizer over all learnable model parameters.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    train_loss_history: list[float] = []
    val_regret_history: list[float] = []
    # Warm-start checkpoint (CPU-cloned weights) so a never-improving run still
    # has a valid best_state to restore. `state_dict()` = {param_name: tensor}.
    best_state: dict = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    best_val_regret = float("inf")   # so the first real value always improves on it
    best_epoch = 0
    epochs_no_improve = 0            # early-stopping counter

    for epoch in range(cfg.max_epochs):
        model.train()        # train mode: dropout/batch-norm updates active
        running_loss = 0.0
        n_seen = 0
        for X_b, tau_b in train_loader:
            # Batch shape (B instances, N vessels, F features); F is ignored ("_").
            B, N, _ = X_b.shape
            X_b = X_b.to(DEVICE)
            tau_b = tau_b.to(DEVICE)

            # Score all B*N vessels in one call, then reshape to (B, N) τ̂ vectors.
            tau_pred = model(X_b.reshape(B * N, -1)).view(B, N)
            optimizer.zero_grad(set_to_none=True)   # clear last step's gradients
            # Differentiable solve: returns (B, N) start times averaged over the
            # n_samples Gaussian perturbations of τ̂. Gradient flows through this.
            starts = p_opt(tau_pred)
            # DFL loss = realised weighted completion under the TRUE τ:
            #   per instance Σ_i w_i*(start_i + tau_true_i), averaged over the batch.
            # As in the blackbox trainer, `starts` are the optimizer's own start
            # times (the smoothed surrogate), not feasibility-re-derived ones.
            loss = (weights_t * (starts + tau_b)).sum(dim=1).mean()
            loss.backward()    # backprop into model parameters
            if cfg.grad_clip > 0:
                # Cap the gradient norm to stabilise these high-variance DFL grads.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                )
            optimizer.step()   # one AdamW parameter update

            # Size-weighted accumulation -> exact per-instance mean even if the
            # final batch is smaller than batch_size.
            running_loss += loss.item() * B
            n_seen += B
        train_loss_history.append(running_loss / max(n_seen, 1))  # avoid /0

        # Held-out decision quality — the metric we actually select on.
        val_regret = _evaluate_regret(
            model, X_inst_val, tau_inst_val, instance, optmodel
        )
        val_regret_history.append(val_regret)
        # `{epoch+1:3d}` pads to width 3; `{...:.2f}` shows 2 decimals.
        print(
            f"  epoch {epoch + 1:3d}: train_loss={train_loss_history[-1]:.2f} "
            f"val_regret={val_regret:.2f}"
        )

        # Early stopping: require a real improvement (the -1e-6 ignores FP noise).
        if val_regret < best_val_regret - 1e-6:
            best_val_regret = val_regret
            best_epoch = epoch
            # Snapshot the improved weights (CPU-cloned so training can't mutate it).
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break   # stop early once patience is exhausted

    # Reload the best-performing weights before returning the trained model.
    model.load_state_dict(best_state)

    return DFLPerturbedResult(
        train_loss_history=train_loss_history,
        val_regret_history=val_regret_history,
        best_epoch=best_epoch,
        best_val_regret=best_val_regret,
        epochs_run=len(train_loss_history),
    )

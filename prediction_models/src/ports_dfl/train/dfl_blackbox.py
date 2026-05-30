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

# `from __future__ import annotations` makes every type hint in this file be
# stored as plain text (not evaluated at import). That lets us write modern
# hints like `int | None` and forward-reference classes on older Pythons.
from __future__ import annotations

# `dataclass` is a decorator (the `@name` line above a class) that auto-writes
# the boilerplate (__init__, __repr__, ==) for a class whose job is to hold data.
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn  # nn = PyTorch's neural-network building blocks (Module, etc.)
# blackboxOpt wraps our MILP so PyTorch can backprop "through" the solver using
# the Pogancic et al. (2020) straight-through interpolation trick.
from pyepo.func import blackboxOpt
# DataLoader batches/shuffles a dataset; TensorDataset wraps tensors as a dataset.
from torch.utils.data import DataLoader, TensorDataset

from ports_dfl.config import DEVICE, SEED, set_seed
from bap_optim.discrete_bap import (
    BAPInstance,
    DiscreteBAP,
    extract_decision,
    schedule_cost_under_true_tau,
)


# `@dataclass` here turns the field declarations below into a config object:
# `DFLBlackboxConfig()` builds one with all the defaults, and you can override
# any field by name, e.g. `DFLBlackboxConfig(lr=5e-4)`.
@dataclass
class DFLBlackboxConfig:
    """Hyperparameters for the blackbox-DFL training loop."""

    # `lr: float = 1e-3` is a field with a TYPE HINT (`float`) and a DEFAULT
    # (`1e-3` = 0.001). The hint is for humans/tools; Python does not enforce it.
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

    # These fields have NO default, so the caller must supply all of them when
    # building the result. `list[float]` means "a list of floats".
    train_loss_history: list[float]   # mean training loss per epoch
    val_regret_history: list[float]   # validation regret per epoch
    best_epoch: int                   # epoch index with the lowest val regret
    best_val_regret: float            # that lowest val regret value
    epochs_run: int                   # how many epochs actually ran (early stop)


# `def name(args) -> ReturnType:` defines a function; the part after `->` is a
# return-type HINT (here `float`). The leading underscore in `_evaluate_regret`
# is a convention meaning "module-private helper" — not part of the public API.
# Each argument has a type hint (`model: nn.Module`, etc.) for readability only.
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
    # Put the network in EVAL mode: turns off dropout / uses running batch-norm
    # stats. Important so evaluation is deterministic and matches deployment.
    model.eval()
    regrets = []
    # `with torch.no_grad():` is a context manager that disables autograd inside
    # the block — no gradient graph is built, saving memory/time. We only need
    # forward predictions here, never a backward pass.
    with torch.no_grad():
        # `zip(...)` pairs each instance's features with its true τ; iterating
        # gives one (x_inst, tau_true) pair at a time. `strict=True` (Python 3.10+)
        # raises if the two arrays differ in length — a cheap guard against a
        # silent off-by-one mismatch between features and labels.
        for x_inst, tau_true in zip(X_inst, tau_inst, strict=True):
            # `torch.as_tensor` wraps the numpy row as a tensor (no copy if it can
            # reuse the buffer) and moves it to the compute DEVICE (cpu/gpu).
            x_t = torch.as_tensor(x_inst, dtype=torch.float32, device=DEVICE)
            # Forward pass -> predicted τ̂. `.detach()` drops autograd tracking,
            # `.cpu()` moves it back to host memory, `.numpy()` converts to numpy,
            # `.ravel()` flattens to a 1-D length-N vector the solver expects.
            tau_pred = model(x_t).detach().cpu().numpy().ravel()

            # --- Predicted decision: solve the MILP under the model's τ̂ ---
            optmodel.setObj(tau_pred)   # push τ̂ into the mutable Pyomo Param
            optmodel.solve()            # solve; result read back via extract_decision
            # Lock in the (assignment, ordering) the solver chose under τ̂.
            assign_pred, order_pred = extract_decision(optmodel)
            # Re-evaluate that SAME decision under the TRUE τ: start times are
            # recomputed feasibly, so this is the real cost of acting on τ̂.
            cost_pred, _ = schedule_cost_under_true_tau(
                assign_pred, order_pred, tau_true, instance.arrivals, instance.weights
            )

            # --- Full-information (FI) benchmark: solve under the TRUE τ ---
            optmodel.setObj(tau_true)
            optmodel.solve()
            assign_fi, order_fi = extract_decision(optmodel)
            # FI cost is the post-hoc optimum (best achievable knowing reality).
            cost_fi, _ = schedule_cost_under_true_tau(
                assign_fi, order_fi, tau_true, instance.arrivals, instance.weights
            )

            # Regret = how much worse our τ̂-driven decision is vs. the optimum.
            # By construction this is >= 0 (FI is the best possible cost).
            regrets.append(cost_pred - cost_fi)
    # Average regret over the validation set; `float(...)` unwraps numpy scalar.
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
    # `config or DFLBlackboxConfig()` uses the passed config if it is truthy,
    # otherwise builds one with all defaults. (`None` is falsy, so `config=None`
    # picks the default object.)
    cfg = config or DFLBlackboxConfig()
    set_seed(cfg.seed)         # reseed python/numpy/torch RNGs for reproducibility
    model = model.to(DEVICE)   # move all model weights onto the compute device
    # Build the MILP optimizer ONCE. We re-solve it under different τ rather than
    # rebuilding it each step — its τ Param is mutable (see discrete_bap.py).
    optmodel = DiscreteBAP(instance)
    # Wrap the solver so it's differentiable. `lambd` is the interpolation
    # strength: larger = smoother (more informative) but more biased gradients.
    bb_opt = blackboxOpt(
        optmodel, lambd=cfg.blackbox_lambd, processes=cfg.processes
    )

    # Pre-convert the (constant) priority weights to a device tensor so the loss
    # can multiply them against predicted-schedule completion times each step.
    weights_t = torch.as_tensor(instance.weights, dtype=torch.float32, device=DEVICE)

    # Bundle features + true τ into a dataset. TensorDataset returns aligned rows
    # (X[k], tau[k]); the DataLoader then yields shuffled mini-batches of them.
    train_ds = TensorDataset(
        torch.as_tensor(X_inst_train, dtype=torch.float32),
        torch.as_tensor(tau_inst_train, dtype=torch.float32),
    )
    # `shuffle=True` reorders instances each epoch (less overfitting to order).
    # `num_workers=0` loads in the main process — simplest, avoids fork issues
    # given the Pyomo/Gurobi solver state held by the optimizer.
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )

    # AdamW = Adam optimizer with decoupled weight decay (L2 regularisation).
    # `model.parameters()` hands it every learnable tensor to update.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    train_loss_history: list[float] = []
    val_regret_history: list[float] = []
    # Initialize best_state to the warm-start so a never-improves run still
    # restores a valid checkpoint (rather than leaving best_state=None).
    # `state_dict()` is a dict {param_name: tensor} of the model's weights;
    # the dict comprehension copies each tensor to CPU (`.cpu()`), detaches it
    # from autograd, and `.clone()`s it so later training steps can't mutate the
    # saved snapshot in place.
    best_state: dict = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    best_val_regret = float("inf")   # +infinity so the first epoch always "wins"
    best_epoch = 0
    epochs_no_improve = 0            # counter that drives early stopping

    # One pass over the whole training set = one "epoch".
    for epoch in range(cfg.max_epochs):
        model.train()        # TRAIN mode: enable dropout / batch-norm updates
        running_loss = 0.0   # sum of (loss * batch_size) used to average later
        n_seen = 0           # total instances seen this epoch (the denominator)
        for X_b, tau_b in train_loader:
            # Unpack the batch shape: B instances, N vessels, F features (F is the
            # "_" we ignore here). Each batch is a 3-D tensor (B, N, F).
            B, N, _ = X_b.shape
            X_b = X_b.to(DEVICE)
            tau_b = tau_b.to(DEVICE)

            # Flatten to (B*N, F) so the per-vessel regressor scores every vessel
            # in one forward call, then `.view(B, N)` reshapes the predictions
            # back to one τ̂ vector per instance.
            tau_pred = model(X_b.reshape(B * N, -1)).view(B, N)
            # Clear gradients from the previous step. `set_to_none=True` frees the
            # grad tensors (slightly faster / less memory) instead of zeroing them.
            optimizer.zero_grad(set_to_none=True)
            # Solve DBAP under predicted τ. Returns shape (B, N) of start times.
            # `bb_opt` is differentiable, so `starts` carries gradient info.
            starts = bb_opt(tau_pred)
            # DFL loss = realised weighted completion time under the TRUE τ:
            #   per instance  Σ_i w_i * (start_i + tau_true_i),  then averaged.
            # `starts + tau_b` = completion time per vessel; multiply by weights;
            # `.sum(dim=1)` collapses the N-vessel axis -> one cost per instance;
            # `.mean()` averages over the B instances in the batch.
            # NOTE: `starts` here are the optimizer's start times under τ̂ (the
            # differentiable surrogate), NOT the feasibility-re-derived starts used
            # for the val regret metric. That asymmetry is intentional in PyEPO:
            # the gradient must flow through the solver's own output.
            loss = (weights_t * (starts + tau_b)).sum(dim=1).mean()
            loss.backward()   # backprop: fills each parameter's `.grad`
            if cfg.grad_clip > 0:
                # Rescale gradients so their global norm <= grad_clip; tames the
                # noisy/large gradients that blackbox differentiation can produce.
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()  # apply one AdamW update using the (clipped) grads

            # Weight each batch's mean loss by its size so the epoch average is a
            # true per-instance mean even if the last batch is smaller.
            running_loss += loss.item() * B
            n_seen += B
        # `max(n_seen, 1)` guards against divide-by-zero on an empty loader.
        train_loss_history.append(running_loss / max(n_seen, 1))

        # Decision-quality metric on held-out data: the true selection signal.
        val_regret = _evaluate_regret(
            model, X_inst_val, tau_inst_val, instance, optmodel
        )
        val_regret_history.append(val_regret)
        # f-string with format specs: `{epoch+1:3d}` pads the epoch to width 3,
        # `{...:.2f}` prints 2 decimals. `train_loss_history[-1]` is the last item.
        print(
            f"  epoch {epoch + 1:3d}: train_loss={train_loss_history[-1]:.2f} "
            f"val_regret={val_regret:.2f}"
        )

        # Early stopping on validation regret. The `- 1e-6` requires a tiny but
        # real improvement, so floating-point noise doesn't count as "better".
        if val_regret < best_val_regret - 1e-6:
            best_val_regret = val_regret
            best_epoch = epoch
            # Snapshot the improved weights (same CPU-clone pattern as above).
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            # No improvement: count it; bail out once we've waited `patience` epochs.
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    # Restore the best checkpoint so the returned model is the one that
    # generalised best, not whatever the last (possibly overfit) epoch produced.
    model.load_state_dict(best_state)

    return DFLBlackboxResult(
        train_loss_history=train_loss_history,
        val_regret_history=val_regret_history,
        best_epoch=best_epoch,
        best_val_regret=best_val_regret,
        epochs_run=len(train_loss_history),
    )

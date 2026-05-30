"""End-to-end DFL on a real Discrete Berth Allocation Problem (DBAP).

Builds multi-berth instances of N vessels, trains a linear predictor under
both predict-then-optimize (MSE) and decision-focused (blackbox-DFL) losses,
and reports a full battery of predictive + decision-quality metrics.

Pipeline
--------
1. Load the cleaned training set, encode features, take fold 0 as train/val.
2. Reshape rows into instances of N vessels (a single planning horizon).
3. Generate one synthetic instance descriptor (arrivals + weights) shared
   across all instances of the same fold.
4. Pre-compute full-information (FI) schedules under ground-truth τ for every
   instance — the post-hoc optimal benchmark used for regret comparisons.
5. Train the PtO baseline with MSE.
6. Train the DFL model with blackbox differentiation through the DBAP MILP.
7. Evaluate both on held-out instances:
     - predictive: MAE, RMSE, MAPE
     - decision quality: weighted completion time under prediction-driven schedule
     - decision quality: regret vs FI optimum
     - schedule structure: makespan, berth utilization, mean wait time
     - assignment overlap: % of vessels assigned to the same berth as in the FI decision

Usage:
    python scripts/run_dfl_real_bap.py [--n_vessels 6 --n_berths 2 ...]
"""

# `from __future__ import annotations` makes all type hints (e.g. `str | None`)
# be treated as plain strings, so newer syntax works on older Python versions.
from __future__ import annotations

import argparse  # parses command-line flags like --n_vessels
import json
import sys
import time
import warnings
from pathlib import Path  # object-oriented file paths (works cross-platform)

import numpy as np
import pandas as pd
import torch

# Hide noisy UserWarnings (e.g. from sklearn/torch) so the console stays readable.
warnings.filterwarnings("ignore", category=UserWarning)
# Add the repo's `src/` folder to Python's import search path so the
# `ports_dfl` package can be imported even when running this script directly.
# `Path(__file__).resolve().parents[1]` = the directory two levels up from this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# The optimizers (bap_optim) now live in the sibling top-level package optimizers/src.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "optimizers" / "src"))

import torch.nn as nn

from ports_dfl.config import DEVICE, RESULTS_DIR, SEED
from ports_dfl.data.encoders import build_preprocessor
from ports_dfl.data.loader import load_training_dataset, split_features_target
from ports_dfl.data.splits import make_cv_splits
from ports_dfl.metrics.regression import all_metrics, mae
from ports_dfl.models.linear import _LinearHead


class MLPHead(nn.Module):
    """Small MLP regressor: input → hidden → 1.

    Used as the predictor in both PtO and DFL arms so the comparison is
    architecture-controlled.
    """

    # `__init__` is the constructor; `hidden_dim: int = 64` gives a default value.
    # `-> None` is a type hint meaning the method returns nothing.
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()  # required: initialise the parent nn.Module
        # nn.Sequential chains layers; data flows top-to-bottom through them.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # `forward` defines what happens when you call the model like `model(x)`.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_predictor(name: str, n_features: int) -> nn.Module:
    """Build a fresh predictor module by name."""
    if name == "linear":
        return _LinearHead(n_features)
    if name == "mlp":
        return MLPHead(n_features)
    # f-string with `!r` inserts the repr of `name` (quotes shown) into the message.
    raise ValueError(f"Unknown predictor: {name!r}")
from bap_optim.discrete_bap import (
    DiscreteBAP,
    extract_decision,
    generate_bap_instance,
    schedule_cost_under_true_tau,
)
from ports_dfl.train.dfl_blackbox import (
    DFLBlackboxConfig,
    train_dfl_blackbox,
)
from ports_dfl.train.pto import TrainConfig, predict_pto, train_pto


def _build_instances(
    X: np.ndarray, y: np.ndarray, n_vessels: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into ``(n_inst, n_vessels)`` planning instances."""
    # Seeded random generator -> shuffling is reproducible run-to-run.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))  # random ordering of all row indices
    n_inst = len(X) // n_vessels  # integer division: how many full instances fit
    keep = n_inst * n_vessels  # drop leftover rows that can't fill a whole instance
    # `reshape` groups the kept rows into (n_inst, n_vessels, n_features) blocks.
    X_perm = X[perm[:keep]].reshape(n_inst, n_vessels, X.shape[1])
    y_perm = y[perm[:keep]].reshape(n_inst, n_vessels)
    return X_perm, y_perm


def _berth_utilization(
    starts: np.ndarray, tau: np.ndarray, x_assign: np.ndarray, horizon: float
) -> float:
    """Fraction of horizon during which any berth is busy.

    Uses per-berth occupied intervals; horizon is fixed by the instance
    descriptor used to generate arrivals.
    """
    n_berths = x_assign.shape[1]  # number of columns = number of berths
    busy_per_berth = np.zeros(n_berths)  # accumulator: total busy time per berth
    for i in range(len(starts)):
        for b in range(n_berths):
            # x_assign is a 0/1 matrix; >0.5 treats it as "vessel i uses berth b".
            if x_assign[i, b] > 0.5:
                busy_per_berth[b] += tau[i]  # add this vessel's service time
    # Average busy time across berths, as a fraction of the planning horizon.
    return float(busy_per_berth.mean() / horizon)


def _mean_wait(starts: np.ndarray, arrivals: np.ndarray) -> float:
    """Mean (start - arrival) across vessels."""
    # np.maximum(..., 0.0) clamps negatives to 0 so early starts aren't "negative wait".
    return float(np.maximum(starts - arrivals, 0.0).mean())


def _evaluate_decisions(
    model: torch.nn.Module,
    X_inst: np.ndarray,
    tau_inst: np.ndarray,
    instance,
    optmodel: DiscreteBAP,
    horizon: float,
    tag: str,
) -> tuple[dict, pd.DataFrame]:  # FIX: was `-> dict`; function returns (summary, df)
    """Run the optimizer under model predictions and report decision quality.

    Crucially, the decision (x, z) is re-evaluated under TRUE τ to recompute
    feasible start times. This guarantees regret ≥ 0 because the
    full-information (FI) decision — solved with true τ inside the MILP —
    is by definition optimal under true τ.
    """
    model.eval()  # put model in evaluation mode (disables dropout, etc.)
    rows: list[dict] = []  # one result dict per instance; the `: list[dict]` is a hint
    overlap_pct: list[float] = []
    # `torch.no_grad()` disables gradient tracking -> faster, less memory for inference.
    with torch.no_grad():
        # `zip(..., strict=True)` pairs items and errors if lengths differ.
        for x_inst, tau_true in zip(X_inst, tau_inst, strict=True):
            # Convert this instance's features to a torch tensor on the right device.
            x_t = torch.as_tensor(x_inst, dtype=torch.float32, device=DEVICE)
            # Predict τ, then move back to CPU/numpy; `.ravel()` flattens to 1-D.
            tau_pred = model(x_t).detach().cpu().numpy().ravel()

            # Decision under predicted τ, evaluated feasibly under true τ
            optmodel.setObj(tau_pred)  # feed predicted service times into the MILP
            optmodel.solve()  # solve for the optimal assignment + ordering
            assign_pred, order_pred = extract_decision(optmodel)
            cost_pred, starts_pred = schedule_cost_under_true_tau(
                assign_pred, order_pred, tau_true, instance.arrivals, instance.weights
            )

            # Full-information (FI) decision: solve under true τ
            optmodel.setObj(tau_true)
            optmodel.solve()
            assign_fi, order_fi = extract_decision(optmodel)
            cost_fi, starts_fi = schedule_cost_under_true_tau(
                assign_fi, order_fi, tau_true, instance.arrivals, instance.weights
            )

            rows.append(
                {
                    "true_cost_pred_decision": cost_pred,
                    "true_cost_fi_decision": cost_fi,
                    # Regret = how much worse the prediction-driven plan is vs the optimum.
                    "regret": cost_pred - cost_fi,
                    "makespan_pred": float((starts_pred + tau_true).max()),
                    "makespan_fi": float((starts_fi + tau_true).max()),
                    "wait_pred": _mean_wait(starts_pred, instance.arrivals),
                    "wait_fi": _mean_wait(starts_fi, instance.arrivals),
                    "util_pred": _berth_utilization(starts_pred, tau_true, assign_pred, horizon),
                    "util_fi": _berth_utilization(starts_fi, tau_true, assign_fi, horizon),
                }
            )

            # Fraction of vessels assigned to the same berth as the FI decision.
            # argmax(axis=1) -> chosen berth index per vessel; == compares element-wise.
            overlap = float((assign_pred.argmax(axis=1) == assign_fi.argmax(axis=1)).mean())
            overlap_pct.append(overlap)

    # Turn the list of per-instance dicts into a table (one row per instance).
    df = pd.DataFrame(rows)
    # `summary` aggregates the table into single mean values for reporting.
    summary = {
        "model": tag,
        "weighted_cost_pred_decision_mean": df["true_cost_pred_decision"].mean(),
        "weighted_cost_fi_mean": df["true_cost_fi_decision"].mean(),
        "regret_mean": df["regret"].mean(),
        # Mean regret as a % of the optimal cost (lower is better).
        # NOTE: if the FI cost mean is 0 this divides by zero -> inf/nan (see review).
        "regret_relative_pct": 100.0 * df["regret"].mean() / df["true_cost_fi_decision"].mean(),
        "makespan_pred_mean": df["makespan_pred"].mean(),
        "makespan_fi_mean": df["makespan_fi"].mean(),
        "mean_wait_pred": df["wait_pred"].mean(),
        "mean_wait_fi": df["wait_fi"].mean(),
        "berth_utilization_pred": df["util_pred"].mean(),
        "berth_utilization_fi": df["util_fi"].mean(),
        "fi_assignment_overlap_pct": 100.0 * float(np.mean(overlap_pct)),
    }
    return summary, df


def main() -> None:
    # argparse reads command-line flags. Each add_argument defines one flag:
    #   --name, the value type, and a default used when the flag is omitted.
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_vessels", type=int, default=6)
    parser.add_argument("--n_berths", type=int, default=2)
    parser.add_argument("--horizon", type=float, default=120.0)
    parser.add_argument("--n_train_instances", type=int, default=80)
    parser.add_argument("--n_val_instances", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    # `help=...` text shows up when the script is run with -h/--help.
    parser.add_argument("--lr", type=float, default=5e-3, help="DFL learning rate.")
    parser.add_argument("--pto_lr", type=float, default=1e-2, help="PtO learning rate (often higher).")
    parser.add_argument("--blackbox_lambd", type=float, default=10.0)
    parser.add_argument(
        # `choices=[...]` restricts the value to the listed options.
        "--predictor", choices=["linear", "mlp"], default="linear",
        help="Architecture for both PtO and DFL arms (controlled comparison)."
    )
    args = parser.parse_args()  # read flags from the command line into `args`

    print("Loading dataset...")
    df = load_training_dataset()
    X, y = split_features_target(df)  # split columns into features X and target y
    splits = make_cv_splits(df)  # list of (train_idx, val_idx) cross-validation folds
    train_idx, val_idx = splits[0]  # use fold 0 only

    pre = build_preprocessor(categorical_strategy="target")
    # `.iloc[idx]` selects rows by position; `.to_numpy()` converts to a numpy array.
    y_train = y.iloc[train_idx].to_numpy()
    y_val = y.iloc[val_idx].to_numpy()
    # `fit_transform` learns the encoding from TRAIN data, then applies it;
    # for val we only `transform` (no re-fitting) to avoid data leakage.
    X_train = pre.fit_transform(X.iloc[train_idx], y_train).astype(np.float32)
    X_val = pre.transform(X.iloc[val_idx]).astype(np.float32)
    n_features = X_train.shape[1]  # number of feature columns after encoding

    instance = generate_bap_instance(
        n_vessels=args.n_vessels,
        n_berths=args.n_berths,
        horizon_hours=args.horizon,
        seed=SEED,
    )
    # f-strings (the f"..." prefix) insert variable values via {curly braces}.
    print(f"\nDBAP instance: N={args.n_vessels}, M={args.n_berths}, horizon={args.horizon}h")
    print(f"  arrivals: {instance.arrivals.round(1)}")
    print(f"  weights:  {instance.weights.round(2)}")

    # --- Build instances ---------------------------------------------------
    # Use different seeds for train vs val so the two instance sets differ.
    X_inst_train, y_inst_train = _build_instances(X_train, y_train, args.n_vessels, SEED)
    X_inst_val, y_inst_val = _build_instances(X_val, y_val, args.n_vessels, SEED + 1)

    # `[: n]` slices the first n instances (caps how many we train/evaluate on).
    X_inst_train = X_inst_train[: args.n_train_instances]
    y_inst_train = y_inst_train[: args.n_train_instances]
    X_inst_val = X_inst_val[: args.n_val_instances]
    y_inst_val = y_inst_val[: args.n_val_instances]
    print(f"  train inst: {X_inst_train.shape[0]} | val inst: {X_inst_val.shape[0]}\n")

    out_dir = RESULTS_DIR / "dfl_real_bap"  # `/` joins paths with pathlib
    out_dir.mkdir(parents=True, exist_ok=True)  # create folder (and parents) if missing

    optmodel = DiscreteBAP(instance)

    # --- PtO baseline ------------------------------------------------------
    print("=" * 70)
    print("Training PtO baseline (MSE loss)")
    print("=" * 70)
    # `.to(DEVICE)` moves the model to CPU or GPU as configured.
    pto_module = make_predictor(args.predictor, n_features).to(DEVICE)
    t0 = time.time()  # start a wall-clock timer to measure training duration
    train_pto(
        pto_module,
        X_train, y_train, X_val, y_val,
        TrainConfig(
            lr=args.pto_lr, weight_decay=1e-3,
            batch_size=256, max_epochs=200, patience=20,
        ),
    )
    pto_train_time = time.time() - t0  # elapsed seconds since t0
    pto_preds = predict_pto(pto_module, X_val)  # predictions on the validation set
    pto_pred_metrics = all_metrics(y_val, pto_preds)  # MAE/RMSE/MAPE dict
    print(f"  PtO MSE training time: {pto_train_time:.1f}s")
    print(f"  PtO val MAE: {pto_pred_metrics['mae']:.3f}h | "
          f"RMSE: {pto_pred_metrics['rmse']:.3f}h | "
          f"MAPE: {pto_pred_metrics['mape']:.3f}\n")

    print("Evaluating PtO decisions on val instances...")
    pto_decision, pto_decision_df = _evaluate_decisions(
        pto_module, X_inst_val, y_inst_val, instance, optmodel, args.horizon, "PtO (MSE)"
    )

    # --- DFL training ------------------------------------------------------
    print("=" * 70)
    print("Training DFL (blackbox)")
    print("=" * 70)
    dfl_module = make_predictor(args.predictor, n_features).to(DEVICE)
    # Copy the trained PtO weights into the DFL model so DFL starts from a good point.
    dfl_module.load_state_dict(pto_module.state_dict())  # warm-start from PtO

    cfg = DFLBlackboxConfig(
        lr=args.lr,
        weight_decay=1e-4,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=8,
        blackbox_lambd=args.blackbox_lambd,
        processes=1,
    )
    t0 = time.time()
    result = train_dfl_blackbox(
        dfl_module,
        X_inst_train, y_inst_train,
        X_inst_val, y_inst_val,
        instance,
        cfg,
    )
    dfl_train_time = time.time() - t0
    # Reuse predict_pto just to get raw predictions (it's a generic forward pass).
    dfl_preds = predict_pto(dfl_module, X_val)
    dfl_pred_metrics = all_metrics(y_val, dfl_preds)
    print(f"\n  DFL training time: {dfl_train_time:.1f}s")
    print(f"  DFL val MAE: {dfl_pred_metrics['mae']:.3f}h | "
          f"RMSE: {dfl_pred_metrics['rmse']:.3f}h | "
          f"MAPE: {dfl_pred_metrics['mape']:.3f}\n")

    print("Evaluating DFL decisions on val instances...")
    dfl_decision, dfl_decision_df = _evaluate_decisions(
        dfl_module, X_inst_val, y_inst_val, instance, optmodel, args.horizon, "DFL (blackbox)"
    )

    # --- Summary -----------------------------------------------------------
    print("\n" + "=" * 70)
    print(" Predictive metrics ")
    print("=" * 70)
    # Build a 2-row table: one row per model. `**dict` unpacks the metrics dict's
    # key/value pairs into this dict (so each metric becomes its own column).
    pred_summary = pd.DataFrame(
        [
            {"model": "PtO (MSE)", **pto_pred_metrics, "train_seconds": pto_train_time},
            {"model": "DFL (blackbox)", **dfl_pred_metrics, "train_seconds": dfl_train_time},
        ]
    )
    # `float_format=lambda v: ...` formats every float to 3 decimals when printing.
    print(pred_summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 70)
    print(" Decision-quality metrics on real DBAP ")
    print("=" * 70)
    decision_summary = pd.DataFrame([pto_decision, dfl_decision])
    print(decision_summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # `.to_csv(path, index=False)` writes the table to disk without the row-number column.
    pred_summary.to_csv(out_dir / "predictive_summary.csv", index=False)
    decision_summary.to_csv(out_dir / "decision_summary.csv", index=False)
    pto_decision_df.to_csv(out_dir / "pto_per_instance.csv", index=False)
    dfl_decision_df.to_csv(out_dir / "dfl_per_instance.csv", index=False)
    # Per-epoch training trace: epoch index, loss, and validation regret over time.
    pd.DataFrame(
        {
            "epoch": range(len(result.train_loss_history)),
            "train_loss": result.train_loss_history,
            "val_regret": result.val_regret_history,
        }
    ).to_csv(out_dir / "dfl_training_trace.csv", index=False)
    # `with open(...)` opens the file and auto-closes it when the block ends.
    # `vars(args)` turns the argparse namespace into a plain dict for JSON dumping.
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nResults written to {out_dir}/")
    print(
        "\nInterpretation: a successful DFL run shows DFL's regret < PtO's regret, "
        "even if DFL's MAE is worse. Predicted τ that mis-orders short vs long "
        "service times costs more decision-wise than τ that's uniformly "
        "biased — DFL trains for the former."
    )


# This block runs only when the file is executed directly (not when imported),
# which is the standard Python entry-point idiom.
if __name__ == "__main__":
    main()

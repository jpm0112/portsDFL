"""Train PtO vs DFL on a synthetic, contention-tuned BAP problem.

Mirrors ``run_dfl_real_bap.py`` but pulls instances from
:func:`bap_optim.classic_bap.make_classic_problem` instead of the
real Chilean vessel-calls dataset. Exposes the difficulty knobs
(``--contention``, ``--weight_dist``, ``--n_vessels``, ``--noise_std``,
…) that let us stress-test DFL vs PtO.

Outputs land in ``<out_dir>/`` with the same filenames as the real-data
demo so downstream tooling (Figure 6, summary aggregators) can ingest
either run uniformly.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Path bootstrap so this script runs from anywhere.
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# The optimizers (bap_optim) now live in the sibling top-level package optimizers/src.
_OPTIM_SRC = THIS_DIR.parent.parent / "optimizers" / "src"
if str(_OPTIM_SRC) not in sys.path:
    sys.path.insert(0, str(_OPTIM_SRC))

from ports_dfl.config import DEVICE, RESULTS_DIR, SEED, set_seed
from ports_dfl.metrics.regression import all_metrics
from ports_dfl.models.linear import _LinearHead
from bap_optim.classic_bap import make_classic_problem
from bap_optim.discrete_bap import (
    DiscreteBAP,
    extract_decision,
    schedule_cost_under_true_tau,
)
from ports_dfl.train.dfl_blackbox import (
    DFLBlackboxConfig,
    train_dfl_blackbox,
)
from ports_dfl.train.dfl_perturbed import (
    DFLPerturbedConfig,
    train_dfl_perturbed,
)
from ports_dfl.train.pto import TrainConfig, predict_pto, train_pto


class MLPHead(nn.Module):
    """Small MLP used when ``--predictor mlp``."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_predictor(name: str, n_features: int) -> nn.Module:
    if name == "linear":
        return _LinearHead(n_features)
    if name == "mlp":
        return MLPHead(n_features)
    raise ValueError(f"Unknown predictor: {name!r}")


def _mean_wait(starts: np.ndarray, arrivals: np.ndarray) -> float:
    # Clamp negatives to 0 (a vessel can't wait negative time).
    return float(np.maximum(starts - arrivals, 0.0).mean())


def _evaluate_decisions(
    model: nn.Module,
    X_inst: np.ndarray,
    tau_inst: np.ndarray,
    instance,
    optmodel: DiscreteBAP,
    tag: str,
) -> tuple[dict, pd.DataFrame]:
    """Same decision-quality protocol as the real-data demo."""
    model.eval()
    rows: list[dict] = []
    overlap_pct: list[float] = []
    with torch.no_grad():
        for x_inst, tau_true in zip(X_inst, tau_inst, strict=True):
            x_t = torch.as_tensor(x_inst, dtype=torch.float32, device=DEVICE)
            tau_pred = model(x_t).detach().cpu().numpy().ravel()

            # Solve under PREDICTED service times.
            optmodel.setObj(tau_pred)
            optmodel.solve()
            assign_pred, order_pred = extract_decision(optmodel)
            # Score that decision under the TRUE times (what would really happen).
            cost_pred, starts_pred = schedule_cost_under_true_tau(
                assign_pred, order_pred, tau_true, instance.arrivals,
            )

            # Solve again with full information (true times) for comparison.
            optmodel.setObj(tau_true)
            optmodel.solve()
            assign_fi, order_fi = extract_decision(optmodel)
            cost_fi, starts_fi = schedule_cost_under_true_tau(
                assign_fi, order_fi, tau_true, instance.arrivals,
            )

            rows.append(
                {
                    "true_cost_pred_decision": cost_pred,
                    "true_cost_fi_decision": cost_fi,
                    # how much worse the predicted plan is vs the ideal plan
                    "regret": cost_pred - cost_fi,
                    "makespan_pred": float((starts_pred + tau_true).max()),
                    "makespan_fi": float((starts_fi + tau_true).max()),
                    "wait_pred": _mean_wait(starts_pred, instance.arrivals),
                    "wait_fi": _mean_wait(starts_fi, instance.arrivals),
                }
            )
            # Fraction of vessels assigned to the same berth as the FI decision.
            overlap = float(
                (assign_pred.argmax(axis=1) == assign_fi.argmax(axis=1)).mean()
            )
            overlap_pct.append(overlap)

    df = pd.DataFrame(rows)
    fi_mean = df["true_cost_fi_decision"].mean()
    summary = {
        "model": tag,
        "cost_pred_decision_mean": df["true_cost_pred_decision"].mean(),
        "cost_fi_mean": fi_mean,
        "regret_mean": df["regret"].mean(),
        # Guard the FI-mean==0 degenerate case (would be inf/nan otherwise).
        "regret_relative_pct": (
            100.0 * df["regret"].mean() / fi_mean if fi_mean else float("nan")
        ),
        "regret_median": float(df["regret"].median()),
        "regret_p90": float(df["regret"].quantile(0.90)),
        "makespan_pred_mean": df["makespan_pred"].mean(),
        "makespan_fi_mean": df["makespan_fi"].mean(),
        "mean_wait_pred": df["wait_pred"].mean(),
        "mean_wait_fi": df["wait_fi"].mean(),
        "fi_assignment_overlap_pct": 100.0 * float(np.mean(overlap_pct)),
    }
    return summary, df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_vessels", type=int, default=8)
    parser.add_argument("--n_berths", type=int, default=3)
    parser.add_argument("--n_train_instances", type=int, default=80)
    parser.add_argument("--n_val_instances", type=int, default=60)
    parser.add_argument("--max_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3, help="DFL lr.")
    parser.add_argument("--pto_lr", type=float, default=1e-2)
    parser.add_argument("--blackbox_lambd", type=float, default=5.0)
    parser.add_argument(
        "--method", choices=["blackbox", "perturbed"], default="blackbox",
        help="DFL surrogate gradient: 'blackbox' (Pogancic 2020 STE) "
             "or 'perturbed' (Berthet 2020 Gaussian-perturbed argmin).",
    )
    parser.add_argument(
        "--perturbed_sigma", type=float, default=1.0,
        help="Std of the Gaussian perturbation (Berthet method only).",
    )
    parser.add_argument(
        "--perturbed_samples", type=int, default=10,
        help="Number of perturbation samples per step (Berthet method only).",
    )
    parser.add_argument(
        "--predictor", choices=["linear", "mlp"], default="linear",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--contention", type=float, default=1.0)
    parser.add_argument(
        "--weight_dist", choices=["three_class", "uniform", "lognormal"],
        default="three_class",
    )
    parser.add_argument(
        "--arrival", choices=["uniform", "poisson"], default="uniform",
    )
    parser.add_argument("--tau_mean", type=float, default=10.0)
    parser.add_argument("--tau_sigma", type=float, default=0.5)
    parser.add_argument("--noise_std", type=float, default=0.4)
    parser.add_argument("--n_noise_features", type=int, default=8)
    parser.add_argument(
        "--no_weight_feature", action="store_true",
        help="Hide the priority weight from the predictor.",
    )
    # default=None lets us detect "user didn't pass this" and auto-build a name later.
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Where to write results. Default: results/dfl_synthetic_<auto>.",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Short tag added to the auto-generated out_dir name.",
    )
    args = parser.parse_args()

    set_seed(args.seed)  # fix all RNGs so results are reproducible

    print("Building synthetic problem...")
    # Generate features X, true service times tau, and the underlying BAP instance.
    prob = make_classic_problem(
        n_vessels=args.n_vessels,
        n_berths=args.n_berths,
        n_train=args.n_train_instances,
        n_val=args.n_val_instances,
        contention=args.contention,
        weight_dist=args.weight_dist,
        arrival=args.arrival,
        tau_mean=args.tau_mean,
        tau_sigma=args.tau_sigma,
        noise_std=args.noise_std,
        n_noise_features=args.n_noise_features,
        # The flag HIDES the weight, so invert it to decide whether to INCLUDE it.
        include_weight_feature=not args.no_weight_feature,
        seed=args.seed,
    )

    inst = prob.instance  # the fixed problem (arrivals, weights) shared across instances
    print(
        f"  N={args.n_vessels}, M={args.n_berths}, "
        f"horizon={prob.meta['horizon']:.1f}h, "
        f"contention={args.contention:.2f}"
    )
    print(f"  arrivals: {inst.arrivals.round(1)}")
    print(f"  weights:  {inst.weights.round(2)}")
    print(f"  features: {prob.feature_names}")
    print(
        f"  train inst: {prob.X_train.shape[0]} | "
        f"val inst: {prob.X_val.shape[0]}"
    )

    # Flatten train/val for PtO MSE training (per-vessel rows): X is shaped
    # (instances, vessels, features); stack all vessels into one 2-D table.
    X_train_flat = prob.X_train.reshape(-1, prob.X_train.shape[-1])
    y_train_flat = prob.tau_train.reshape(-1)
    X_val_flat = prob.X_val.reshape(-1, prob.X_val.shape[-1])
    y_val_flat = prob.tau_val.reshape(-1)
    n_features = X_train_flat.shape[1]

    if args.out_dir is None:
        tag = f"_{args.tag}" if args.tag else ""
        # Encode the method's key hyperparameters into the folder name so runs
        # don't overwrite each other.
        if args.method == "blackbox":
            method_part = f"bb_l{args.blackbox_lambd:g}"
        else:
            method_part = f"pert_sig{args.perturbed_sigma:g}_k{args.perturbed_samples}"
        args.out_dir = str(
            RESULTS_DIR
            / "dfl_synthetic"
            / (
                f"c{args.contention:.2f}_{method_part}"
                f"_{args.predictor}_s{args.seed}{tag}"
            )
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    optmodel = DiscreteBAP(inst)

    # --- PtO -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Training PtO baseline (MSE loss)")
    print("=" * 70)
    pto_module = make_predictor(args.predictor, n_features).to(DEVICE)
    t0 = time.time()
    # Predict-then-Optimize baseline: train on plain MSE, ignoring downstream cost.
    train_pto(
        pto_module,
        X_train_flat, y_train_flat, X_val_flat, y_val_flat,
        TrainConfig(
            lr=args.pto_lr, weight_decay=1e-3,
            batch_size=256, max_epochs=200, patience=20,
            seed=args.seed,
        ),
    )
    pto_train_time = time.time() - t0
    pto_preds = predict_pto(pto_module, X_val_flat)
    pto_pred = all_metrics(y_val_flat, pto_preds)
    print(
        f"  PtO MAE: {pto_pred['mae']:.3f}h  RMSE: {pto_pred['rmse']:.3f}h"
        f"  ({pto_train_time:.1f}s)"
    )

    print("Evaluating PtO decisions...")
    pto_decision, pto_df = _evaluate_decisions(
        pto_module, prob.X_val, prob.tau_val, inst, optmodel, "PtO (MSE)"
    )

    # --- DFL -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Training DFL (blackbox)")
    print("=" * 70)
    dfl_module = make_predictor(args.predictor, n_features).to(DEVICE)
    # Warm-start DFL from the trained PtO baseline instead of from scratch.
    dfl_module.load_state_dict(pto_module.state_dict())

    t0 = time.time()
    if args.method == "blackbox":
        cfg = DFLBlackboxConfig(
            lr=args.lr,
            weight_decay=1e-4,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=8,
            blackbox_lambd=args.blackbox_lambd,
            processes=1,
            seed=args.seed,
        )
        print(f"  using blackbox method (Pogancic 2020), lambd={args.blackbox_lambd}")
        result = train_dfl_blackbox(
            dfl_module,
            prob.X_train, prob.tau_train,
            prob.X_val, prob.tau_val,
            inst,
            cfg,
        )
    else:
        cfg = DFLPerturbedConfig(
            lr=args.lr,
            weight_decay=1e-4,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=8,
            n_samples=args.perturbed_samples,
            sigma=args.perturbed_sigma,
            processes=1,
            seed=args.seed,
        )
        print(
            f"  using Berthet perturbed method, sigma={args.perturbed_sigma}, "
            f"n_samples={args.perturbed_samples}"
        )
        result = train_dfl_perturbed(
            dfl_module,
            prob.X_train, prob.tau_train,
            prob.X_val, prob.tau_val,
            inst,
            cfg,
        )
    dfl_train_time = time.time() - t0
    # predict_pto is a generic forward pass, reused here for raw predictions.
    dfl_preds = predict_pto(dfl_module, X_val_flat)
    dfl_pred = all_metrics(y_val_flat, dfl_preds)
    print(
        f"\n  DFL MAE: {dfl_pred['mae']:.3f}h  RMSE: {dfl_pred['rmse']:.3f}h"
        f"  ({dfl_train_time:.1f}s)"
    )

    print("Evaluating DFL decisions...")
    dfl_tag = (
        f"DFL ({args.method})"
        if args.method != "blackbox"
        else "DFL (blackbox)"
    )
    dfl_decision, dfl_df = _evaluate_decisions(
        dfl_module, prob.X_val, prob.tau_val, inst, optmodel, dfl_tag
    )

    # --- Output --------------------------------------------------------
    pred_summary = pd.DataFrame(
        [
            {"model": "PtO (MSE)", **pto_pred, "train_seconds": pto_train_time},
            # FIX: was hard-coded "DFL (blackbox)", which mislabeled the row when
            # --method perturbed is used. Use dfl_tag so the label matches the method.
            {"model": dfl_tag, **dfl_pred, "train_seconds": dfl_train_time},
        ]
    )
    decision_summary = pd.DataFrame([pto_decision, dfl_decision])

    pred_summary.to_csv(out_dir / "predictive_summary.csv", index=False)
    decision_summary.to_csv(out_dir / "decision_summary.csv", index=False)
    pto_df.to_csv(out_dir / "pto_per_instance.csv", index=False)
    dfl_df.to_csv(out_dir / "dfl_per_instance.csv", index=False)
    pd.DataFrame(
        {
            "epoch": range(len(result.train_loss_history)),
            "train_loss": result.train_loss_history,
            "val_regret": result.val_regret_history,
        }
    ).to_csv(out_dir / "dfl_training_trace.csv", index=False)
    # Merge args with problem metadata and dump as JSON for reproducibility.
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), **prob.meta}, f, indent=2)

    print("\n" + "=" * 70)
    print(" Summary ")
    print("=" * 70)
    print(decision_summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    pto_m = pto_decision["regret_mean"]
    dfl_m = dfl_decision["regret_mean"]
    pto_p = pto_decision["regret_p90"]   # 90th-percentile (worst-case-ish) regret
    dfl_p = dfl_decision["regret_p90"]
    print(
        f"\n  mean regret: PtO {pto_m:.2f}  DFL {dfl_m:.2f}  "
        f"diff {dfl_m - pto_m:+.2f} ({100*(dfl_m/pto_m - 1):+.1f}%)"
    )
    print(
        f"  p90 regret:  PtO {pto_p:.2f}  DFL {dfl_p:.2f}  "
        f"diff {dfl_p - pto_p:+.2f} ({100*(dfl_p/pto_p - 1):+.1f}%)"
    )
    # Lower regret is better, so DFL "wins" when its regret is below PtO's.
    if dfl_m < pto_m:
        print("  ==> DFL beats PtO on MEAN regret.")
    if dfl_p < pto_p:
        print("  ==> DFL beats PtO on P90 regret.")
    print(f"\n  results: {out_dir}/")


if __name__ == "__main__":
    main()

"""Train PtO vs DFL on a synthetic, contention-tuned BAP problem.

Mirrors ``run_dfl_real_bap.py`` but pulls instances from
:func:`ports_dfl.optim.classic_bap.make_classic_problem` instead of the
real Chilean vessel-calls dataset. Exposes the difficulty knobs
(``--contention``, ``--weight_dist``, ``--n_vessels``, ``--noise_std``,
…) that let us stress-test DFL vs PtO.

Outputs land in ``<out_dir>/`` with the same filenames as the real-data
demo so downstream tooling (Figure 6, summary aggregators) can ingest
either run uniformly.
"""

import argparse  # builds the command-line interface (the --flags below)
import json       # writes the config.json output file
import sys        # lets us tweak sys.path so imports below can be found
import time       # measures how long training takes
from pathlib import Path  # object-oriented file paths (nicer than raw strings)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Path bootstrap so this script runs from anywhere.
# __file__ is this script's path; .resolve() makes it absolute; .parent is its folder.
THIS_DIR = Path(__file__).resolve().parent
# The "src" folder sits next to the "scripts" folder (one level up, then into src).
SRC_DIR = THIS_DIR.parent / "src"
# sys.path is Python's list of folders to search for imports. We add src to the
# FRONT (index 0) so "import ports_dfl..." below works even if you launch from elsewhere.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ports_dfl.config import DEVICE, RESULTS_DIR, SEED, set_seed
from ports_dfl.metrics.regression import all_metrics
from ports_dfl.models.linear import _LinearHead
from ports_dfl.optim.classic_bap import make_classic_problem
from ports_dfl.optim.discrete_bap import (
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


# A class that subclasses nn.Module is a PyTorch neural network. PyTorch will
# track its weights and compute gradients for them automatically.
class MLPHead(nn.Module):
    """Small MLP used when ``--predictor mlp``."""

    # __init__ is the constructor (runs when you create the object). Type hints
    # like "input_dim: int" and "hidden_dim: int = 64" say the expected types;
    # "= 64" gives a default value. "-> None" means this returns nothing.
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()  # MUST call the nn.Module constructor first
        # nn.Sequential chains layers: each one's output feeds the next.
        # Linear = fully-connected layer, ReLU = activation, Dropout = regularization.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # final layer outputs a single number (predicted tau)
        )

    # forward defines the actual computation. PyTorch calls it when you do model(x).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Factory function: returns the right model object given the --predictor choice.
def make_predictor(name: str, n_features: int) -> nn.Module:
    if name == "linear":
        return _LinearHead(n_features)
    if name == "mlp":
        return MLPHead(n_features)
    # f-string with !r shows the value's repr (with quotes), helpful for debugging.
    raise ValueError(f"Unknown predictor: {name!r}")


# Average waiting time = how long each vessel waits between arriving and starting.
def _mean_wait(starts: np.ndarray, arrivals: np.ndarray) -> float:
    # np.maximum(..., 0.0) clamps negatives to 0 (a vessel can't wait negative time).
    # .mean() averages over all vessels; float(...) converts the numpy scalar to a plain float.
    return float(np.maximum(starts - arrivals, 0.0).mean())


def _evaluate_decisions(
    model: nn.Module,
    X_inst: np.ndarray,
    tau_inst: np.ndarray,
    instance,
    optmodel: DiscreteBAP,
    tag: str,
# "-> tuple[dict, pd.DataFrame]" says this returns a 2-item tuple: a summary
# dict and a per-instance table. "instance" has no type hint (its type lives in
# the missing data subpackage), which is fine.
) -> tuple[dict, pd.DataFrame]:
    """Same decision-quality protocol as the real-data demo."""
    model.eval()  # switch model to evaluation mode (turns off Dropout, etc.)
    # Type hints on empty lists tell readers what they'll hold; they're just lists.
    rows: list[dict] = []
    overlap_pct: list[float] = []
    # torch.no_grad() disables gradient tracking: faster + less memory since we're
    # only making predictions here, not training.
    with torch.no_grad():
        # zip pairs up each instance's features with its true service times.
        # strict=True (Python 3.10+) raises if the two have different lengths.
        for x_inst, tau_true in zip(X_inst, tau_inst, strict=True):
            # Convert the numpy feature array into a torch tensor on the right device.
            x_t = torch.as_tensor(x_inst, dtype=torch.float32, device=DEVICE)
            # Predict tau, detach from the graph, move to CPU, to numpy, flatten to 1-D.
            tau_pred = model(x_t).detach().cpu().numpy().ravel()

            # Solve the berth-allocation problem using the PREDICTED service times.
            optmodel.setObj(tau_pred)
            optmodel.solve()
            assign_pred, order_pred = extract_decision(optmodel)
            # Score that decision under the TRUE times (what would really happen).
            cost_pred, starts_pred = schedule_cost_under_true_tau(
                assign_pred, order_pred, tau_true,
                instance.arrivals, instance.weights,
            )

            # Now solve again with perfect info (the true times) for comparison.
            optmodel.setObj(tau_true)
            optmodel.solve()
            assign_fi, order_fi = extract_decision(optmodel)  # "fi" = full information
            cost_fi, starts_fi = schedule_cost_under_true_tau(
                assign_fi, order_fi, tau_true,
                instance.arrivals, instance.weights,
            )

            # Record one row of metrics for this instance.
            rows.append(
                {
                    "true_cost_pred_decision": cost_pred,
                    "true_cost_fi_decision": cost_fi,
                    # regret = how much worse the predicted plan is vs the ideal plan.
                    "regret": cost_pred - cost_fi,
                    # makespan = when the last vessel finishes (start + service time).
                    "makespan_pred": float((starts_pred + tau_true).max()),
                    "makespan_fi": float((starts_fi + tau_true).max()),
                    "wait_pred": _mean_wait(starts_pred, instance.arrivals),
                    "wait_fi": _mean_wait(starts_fi, instance.arrivals),
                }
            )
            # argmax(axis=1) gives each vessel's chosen berth; compare predicted vs
            # full-info choice and .mean() gives the fraction of berths that match.
            overlap = float(
                (assign_pred.argmax(axis=1) == assign_fi.argmax(axis=1)).mean()
            )
            overlap_pct.append(overlap)

    # Build a table where each row is one instance, columns are the metric keys above.
    df = pd.DataFrame(rows)
    # Collapse the per-instance table into a single summary dict (means, quantiles).
    summary = {
        "model": tag,
        "weighted_cost_pred_decision_mean": df["true_cost_pred_decision"].mean(),
        "weighted_cost_fi_mean": df["true_cost_fi_decision"].mean(),
        "regret_mean": df["regret"].mean(),
        "regret_relative_pct": (
            100.0 * df["regret"].mean() / df["true_cost_fi_decision"].mean()
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
    # ArgumentParser collects command-line options. Each add_argument defines one
    # flag: "--n_vessels" becomes args.n_vessels. type= converts the text the user
    # types, default= is used when the flag is omitted, help= shows in --help.
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
    # choices=[...] restricts the value to that list; argparse rejects anything else.
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
    # action="store_true" makes this a boolean flag: present -> True, absent -> False.
    # (No value is typed after it.)
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
    # Read the actual command-line values into "args" (e.g. args.n_vessels).
    args = parser.parse_args()

    set_seed(args.seed)  # fix all RNGs so results are reproducible

    print("Building synthetic problem...")
    # Generate the synthetic dataset: features X, true service times tau, and the
    # underlying BAP "instance" (arrivals/weights). The difficulty knobs come from args.
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
        # "not args.no_weight_feature": the flag HIDES the weight, so invert it to
        # decide whether to INCLUDE it.
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

    # Flatten train/val for PtO MSE training (per-vessel rows).
    # X is shaped (instances, vessels, features); reshape(-1, n_features) stacks all
    # vessels from all instances into one big 2-D table. The -1 means "infer this size".
    X_train_flat = prob.X_train.reshape(-1, prob.X_train.shape[-1])
    y_train_flat = prob.tau_train.reshape(-1)  # matching 1-D vector of true taus
    X_val_flat = prob.X_val.reshape(-1, prob.X_val.shape[-1])
    y_val_flat = prob.tau_val.reshape(-1)
    n_features = X_train_flat.shape[1]  # number of columns = features per vessel

    # Only auto-build an output folder name if the user didn't supply --out_dir.
    if args.out_dir is None:
        # Inline (ternary) if: add "_tag" only when args.tag is non-empty.
        tag = f"_{args.tag}" if args.tag else ""
        # ":g" formats a float compactly (drops trailing zeros). Encode the method's
        # key hyperparameters into the folder name so runs don't overwrite each other.
        if args.method == "blackbox":
            method_part = f"bb_l{args.blackbox_lambd:g}"
        else:
            method_part = f"pert_sig{args.perturbed_sigma:g}_k{args.perturbed_samples}"
        # The "/" operator on Path objects joins path parts (cross-platform).
        args.out_dir = str(
            RESULTS_DIR
            / "dfl_synthetic"
            / (
                f"c{args.contention:.2f}_{method_part}"
                f"_{args.predictor}_s{args.seed}{tag}"
            )
        )
    out_dir = Path(args.out_dir)
    # Create the folder (parents=True makes intermediate folders; exist_ok=True
    # avoids an error if it already exists).
    out_dir.mkdir(parents=True, exist_ok=True)

    optmodel = DiscreteBAP(inst)  # the optimization solver we'll reuse for every solve

    # --- PtO -----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Training PtO baseline (MSE loss)")
    print("=" * 70)
    # .to(DEVICE) moves the model to CPU or GPU as configured.
    pto_module = make_predictor(args.predictor, n_features).to(DEVICE)
    t0 = time.time()  # start the stopwatch for training time
    # Predict-then-Optimize baseline: train the predictor to minimize plain MSE,
    # ignoring the downstream scheduling cost.
    train_pto(
        pto_module,
        X_train_flat, y_train_flat, X_val_flat, y_val_flat,
        TrainConfig(
            lr=args.pto_lr, weight_decay=1e-3,
            batch_size=256, max_epochs=200, patience=20,
            seed=args.seed,
        ),
    )
    pto_train_time = time.time() - t0  # elapsed seconds
    pto_preds = predict_pto(pto_module, X_val_flat)  # predictions on the validation set
    pto_pred = all_metrics(y_val_flat, pto_preds)  # MAE/RMSE etc. vs true taus
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
    # state_dict() is the model's learned weights; loading PtO's weights into the
    # DFL model "warm-starts" it from the trained baseline instead of from scratch.
    dfl_module.load_state_dict(pto_module.state_dict())  # warm-start

    t0 = time.time()
    # Pick the DFL gradient-estimation method based on --method.
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
    # Reuse predict_pto just to get predictions out of the model (method is the same).
    dfl_preds = predict_pto(dfl_module, X_val_flat)
    dfl_pred = all_metrics(y_val_flat, dfl_preds)
    print(
        f"\n  DFL MAE: {dfl_pred['mae']:.3f}h  RMSE: {dfl_pred['rmse']:.3f}h"
        f"  ({dfl_train_time:.1f}s)"
    )

    print("Evaluating DFL decisions...")
    # Build a human-readable label like "DFL (blackbox)" or "DFL (perturbed)".
    dfl_tag = (
        f"DFL ({args.method})"
        if args.method != "blackbox"
        else "DFL (blackbox)"
    )
    dfl_decision, dfl_df = _evaluate_decisions(
        dfl_module, prob.X_val, prob.tau_val, inst, optmodel, dfl_tag
    )

    # --- Output --------------------------------------------------------
    # ** unpacks a dict's key/value pairs into this new dict, so each metric in
    # pto_pred/dfl_pred becomes its own column alongside "model" and "train_seconds".
    pred_summary = pd.DataFrame(
        [
            {"model": "PtO (MSE)", **pto_pred, "train_seconds": pto_train_time},
            # FIX: was hard-coded "DFL (blackbox)", which mislabeled the row when
            # --method perturbed is used. Use dfl_tag so the label matches the method.
            {"model": dfl_tag, **dfl_pred, "train_seconds": dfl_train_time},
        ]
    )
    # Two-row table comparing PtO vs DFL decision quality (regret, makespan, ...).
    decision_summary = pd.DataFrame([pto_decision, dfl_decision])

    # .to_csv writes the table to disk; index=False drops pandas' row-number column.
    pred_summary.to_csv(out_dir / "predictive_summary.csv", index=False)
    decision_summary.to_csv(out_dir / "decision_summary.csv", index=False)
    pto_df.to_csv(out_dir / "pto_per_instance.csv", index=False)
    dfl_df.to_csv(out_dir / "dfl_per_instance.csv", index=False)
    # Per-epoch training curve: range(...) numbers the epochs 0,1,2,... matching the
    # length of the recorded loss/regret histories.
    pd.DataFrame(
        {
            "epoch": range(len(result.train_loss_history)),
            "train_loss": result.train_loss_history,
            "val_regret": result.val_regret_history,
        }
    ).to_csv(out_dir / "dfl_training_trace.csv", index=False)
    # "with open(...)" opens the file and guarantees it's closed afterward, even on
    # error. vars(args) turns the argparse Namespace into a dict; we merge it with the
    # problem metadata and dump it all as JSON for reproducibility.
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), **prob.meta}, f, indent=2)

    print("\n" + "=" * 70)
    print(" Summary ")
    print("=" * 70)
    # to_string prints the whole DataFrame neatly; float_format is a tiny function
    # (a lambda) applied to each number so everything shows 3 decimal places.
    print(decision_summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    # Pull out the headline numbers to compare PtO vs DFL.
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


# This block runs main() only when the file is executed directly (python run_dfl_synthetic.py),
# not when it's imported as a module by another file.
if __name__ == "__main__":
    main()

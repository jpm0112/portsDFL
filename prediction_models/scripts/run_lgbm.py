"""Tune and CV-evaluate the LightGBM benchmark (Predict-then-Optimize, not DFL).

Mirrors run_xgb.py: a stock-defaults run plus an Optuna study (TPE + Hyperband),
then a full-metrics re-evaluation of the best configuration. Per-config round
economy comes from LightGBM's own early stopping inside each fit.

Usage:
    python scripts/run_lgbm.py [--n_trials 30]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

# Quiet library chatter (lightgbm) and Optuna's mixed-categorical UserWarning,
# matching the other ML run scripts (run_realmlp / run_tabm / run_node).
warnings.filterwarnings("ignore", category=UserWarning)

# Make the package importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ports_dfl.config import N_FOLDS, OPTUNA_DB_DIR, RESULTS_DIR, SEED
from ports_dfl.data.encoders import build_preprocessor
from ports_dfl.data.loader import load_training_dataset, split_features_target
from ports_dfl.data.splits import make_cv_splits
from ports_dfl.metrics.regression import all_metrics, summarize_folds
from ports_dfl.models.lgbm import LightGBMRegressorModel
from ports_dfl.tuning.runner import make_objective, run_study, trials_to_dataframe
from ports_dfl.tuning.search_spaces import suggest_lgbm


def _evaluate(name: str, model_kwargs: dict, X, y, splits, out_dir: Path) -> pd.DataFrame:
    """CV-evaluate one configuration with full per-fold metrics."""
    fold_metrics: list[dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx].to_numpy()
        y_val = y.iloc[val_idx].to_numpy()

        # Fresh preprocessor per fold, fit on train only, so val never leaks.
        pre = build_preprocessor(categorical_strategy="target")
        X_train = pre.fit_transform(X_train_raw, y_train).astype(np.float32)
        X_val = pre.transform(X_val_raw).astype(np.float32)

        # Distinct seed per fold so folds are independent draws (merge overrides
        # any random_state in model_kwargs without a duplicate-kwarg error).
        model = LightGBMRegressorModel(
            input_dim=X_train.shape[1],
            **{**model_kwargs, "random_state": SEED + fold_idx},
        )
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        fold_metrics.append(all_metrics(y_val, preds))
        print(
            f"  [{name}] fold {fold_idx + 1}/{len(splits)}: "
            f"MAE={fold_metrics[-1]['mae']:.2f}h"
        )
    summary = summarize_folds(fold_metrics)
    summary.to_csv(out_dir / f"cv_summary_{name}.csv")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--study_name", type=str, default="lgbm")
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_training_dataset()
    X, y = split_features_target(df)
    splits = make_cv_splits(df)
    print(f"  rows={len(df)} | folds={len(splits)}\n")

    out_dir = RESULTS_DIR / "lgbm"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Stock defaults ----------------------------------------------------
    print("Evaluating LightGBM with stock defaults...")
    stock_summary = _evaluate("stock", {}, X, y, splits, out_dir)
    print("Stock CV summary:")
    print(stock_summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print()

    # --- Tuned: TPE + Hyperband -------------------------------------------
    print(f"Running Optuna study (TPE + Hyperband) with {args.n_trials} trials...")
    objective = make_objective(
        factory=LightGBMRegressorModel,
        suggest_fn=suggest_lgbm,
        X=X, y=y, splits=splits,
        report_intermediate=True,  # enable per-fold reporting so the pruner acts
    )
    study = run_study(
        study_name=args.study_name,
        objective=objective,
        n_trials=args.n_trials,
        storage_dir=OPTUNA_DB_DIR,
        seed=SEED,
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=N_FOLDS, reduction_factor=3
        ),
    )
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best mean-fold MAE: {study.best_value:.3f}h")
    print(f"Best params: {study.best_params}\n")

    trials_to_dataframe(study).to_csv(out_dir / "trials.csv", index=False)
    with open(out_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)

    print("Re-evaluating best configuration...")
    # Pass a copy so the helper can't mutate Optuna's best_params.
    tuned_summary = _evaluate("tuned", dict(study.best_params), X, y, splits, out_dir)
    print("\n=== Tuned LightGBM — CV summary ===")
    print(tuned_summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    main()

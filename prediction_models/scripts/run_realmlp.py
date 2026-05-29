"""Tune and CV-evaluate RealMLP (pytabkit's pre-tuned MLP).

Pipeline mirrors run_linear.py:
  1. Load training_dataset.csv.
  2. 5-fold StratifiedKFold by Sitio.
  3. Run two studies:
     a. ``stock`` — RealMLP with all-default hyperparameters (1 trial, no tuning).
     b. ``tuned`` — Optuna over the search space in tuning/search_spaces.suggest_realmlp.
  4. Re-evaluate the best configuration with full per-fold metrics.

Usage:
    python scripts/run_realmlp.py [--n_trials 30]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Quiet pytabkit / Lightning chatter
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("PYTORCH_LIGHTNING_DISABLE_LITLOGGER", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ports_dfl.config import OPTUNA_DB_DIR, RESULTS_DIR, SEED
from ports_dfl.data.encoders import build_preprocessor
from ports_dfl.data.loader import load_training_dataset, split_features_target
from ports_dfl.data.splits import make_cv_splits
from ports_dfl.metrics.regression import all_metrics, summarize_folds
from ports_dfl.models.realmlp import RealMLP
from ports_dfl.tuning.runner import make_objective, run_study, trials_to_dataframe
from ports_dfl.tuning.search_spaces import suggest_realmlp


def _evaluate(name: str, model_kwargs: dict, X, y, splits, out_dir: Path) -> pd.DataFrame:
    """CV-evaluate a single hyperparameter configuration."""
    fold_metrics: list[dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_train = y.iloc[train_idx].to_numpy()
        y_val = y.iloc[val_idx].to_numpy()

        pre = build_preprocessor(categorical_strategy="target")
        X_train = pre.fit_transform(X_train_raw, y_train).astype(np.float32)
        X_val = pre.transform(X_val_raw).astype(np.float32)

        model = RealMLP(input_dim=X_train.shape[1], **model_kwargs)
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
    parser.add_argument("--n_epochs", type=int, default=128)
    parser.add_argument("--study_name", type=str, default="realmlp")
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_training_dataset()
    X, y = split_features_target(df)
    splits = make_cv_splits(df)
    print(f"  rows={len(df)} | folds={len(splits)}\n")

    out_dir = RESULTS_DIR / "realmlp"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Stock RealMLP defaults -------------------------------------------
    print("Evaluating RealMLP with stock TD defaults...")
    stock_summary = _evaluate(
        "stock", {"n_epochs": args.n_epochs}, X, y, splits, out_dir
    )
    print("Stock CV summary:")
    print(stock_summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print()

    # --- Tuned RealMLP -----------------------------------------------------
    print(f"Running Optuna study with {args.n_trials} trials...")

    # Wrap the model factory so n_epochs propagates from CLI.
    def factory(input_dim: int, **hp) -> RealMLP:
        return RealMLP(input_dim=input_dim, n_epochs=args.n_epochs, **hp)

    objective = make_objective(
        factory=factory,
        suggest_fn=suggest_realmlp,
        X=X, y=y, splits=splits,
    )
    study = run_study(
        study_name=args.study_name,
        objective=objective,
        n_trials=args.n_trials,
        storage_dir=OPTUNA_DB_DIR,
        seed=SEED,
    )
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best mean-fold MAE: {study.best_value:.3f}h")
    print(f"Best params: {study.best_params}\n")

    trials_to_dataframe(study).to_csv(out_dir / "trials.csv", index=False)
    with open(out_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)

    print("Re-evaluating best configuration...")
    tuned_summary = _evaluate("tuned", dict(study.best_params), X, y, splits, out_dir)
    print("\n=== Tuned RealMLP — CV summary ===")
    print(tuned_summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    main()

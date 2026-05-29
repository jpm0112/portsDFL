"""Tune and CV-evaluate TabM (parameter-efficient ensemble of MLPs).

Usage:
    python scripts/run_tabm.py [--n_trials 30]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ports_dfl.config import OPTUNA_DB_DIR, RESULTS_DIR, SEED
from ports_dfl.data.encoders import build_preprocessor
from ports_dfl.data.loader import load_training_dataset, split_features_target
from ports_dfl.data.splits import make_cv_splits
from ports_dfl.metrics.regression import all_metrics, summarize_folds
from ports_dfl.models.tabm import TabM
from ports_dfl.tuning.runner import make_objective, run_study, trials_to_dataframe
from ports_dfl.tuning.search_spaces import suggest_tabm


def _evaluate_best(best_params: dict, X, y, splits, max_epochs: int, out_dir: Path) -> pd.DataFrame:
    """Re-run the best configuration with full bookkeeping."""
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_train = y.iloc[train_idx].to_numpy()
        y_val = y.iloc[val_idx].to_numpy()

        pre = build_preprocessor(categorical_strategy="target")
        X_train = pre.fit_transform(X_train_raw, y_train).astype(np.float32)
        X_val = pre.transform(X_val_raw).astype(np.float32)

        model = TabM(input_dim=X_train.shape[1], max_epochs=max_epochs, **best_params)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        fold_metrics.append(all_metrics(y_val, preds))
        print(f"  fold {fold_idx + 1}/{len(splits)}: MAE={fold_metrics[-1]['mae']:.2f}h")
    summary = summarize_folds(fold_metrics)
    summary.to_csv(out_dir / "cv_summary.csv")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--max_epochs", type=int, default=128)
    parser.add_argument("--study_name", type=str, default="tabm")
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_training_dataset()
    X, y = split_features_target(df)
    splits = make_cv_splits(df)
    print(f"  rows={len(df)} | folds={len(splits)}\n")

    out_dir = RESULTS_DIR / "tabm"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Optuna study with {args.n_trials} trials...")

    def factory(input_dim: int, **hp) -> TabM:
        return TabM(input_dim=input_dim, max_epochs=args.max_epochs, **hp)

    objective = make_objective(
        factory=factory,
        suggest_fn=suggest_tabm,
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
    summary = _evaluate_best(dict(study.best_params), X, y, splits, args.max_epochs, out_dir)
    print("\n=== TabM — best-config CV summary ===")
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    main()

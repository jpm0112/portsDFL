"""Tune and CV-evaluate the Ridge linear regressor.

Pipeline:
  1. Load training_dataset.csv.
  2. Build a 5-fold StratifiedKFold split on `Sitio`.
  3. Run Optuna (50 trials by default) over a search space that covers
     L2 strength, learning rate, batch size, and feature encoding.
  4. Re-evaluate the best configuration with full per-fold metrics and
     write results/linear/{cv_summary, trials, best_config}.csv.
  5. Sanity-check the PyTorch Ridge against scikit-learn's closed-form
     Ridge to confirm the framework is wired correctly.

Usage:
    python scripts/run_linear.py [--n_trials 50]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import Ridge as SklearnRidge

# Make the package importable when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ports_dfl.config import OPTUNA_DB_DIR, RESULTS_DIR, SEED
from ports_dfl.data.encoders import build_preprocessor
from ports_dfl.data.loader import load_training_dataset, split_features_target
from ports_dfl.data.splits import make_cv_splits
from ports_dfl.metrics.regression import all_metrics, summarize_folds
from ports_dfl.models.linear import LinearRegressor
from ports_dfl.tuning.runner import make_objective, run_study, trials_to_dataframe
from ports_dfl.tuning.search_spaces import suggest_linear


def _suggest_extras(trial: optuna.Trial) -> dict:
    """Tunable preprocessing choices for the linear model."""
    return {
        "categorical_strategy": trial.suggest_categorical(
            "categorical_strategy", ["target", "onehot"]
        ),
        "numeric_scaler": trial.suggest_categorical(
            "numeric_scaler", ["standard", "robust"]
        ),
    }


def _evaluate_best(
    best_params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    splits,
    out_dir: Path,
) -> pd.DataFrame:
    """Re-run the best configuration with full bookkeeping."""
    cat_strategy = best_params.pop("categorical_strategy", "target")
    scaler = best_params.pop("numeric_scaler", "standard")

    fold_metrics = []
    sklearn_metrics = []  # closed-form sanity check
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx].to_numpy()
        y_val = y.iloc[val_idx].to_numpy()

        pre = build_preprocessor(
            categorical_strategy=cat_strategy, numeric_scaler=scaler
        )
        X_train = pre.fit_transform(X_train_raw, y_train).astype(np.float32)
        X_val = pre.transform(X_val_raw).astype(np.float32)

        # PyTorch Ridge
        model = LinearRegressor(input_dim=X_train.shape[1], **best_params)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        fold_metrics.append(all_metrics(y_val, preds))

        # Closed-form Ridge baseline (sanity check)
        ridge_alpha = max(
            best_params.get("weight_decay", 1e-3) * len(X_train), 1e-6
        )
        sk = SklearnRidge(alpha=ridge_alpha, random_state=SEED).fit(X_train, y_train)
        sklearn_metrics.append(all_metrics(y_val, sk.predict(X_val)))

        print(
            f"  fold {fold_idx + 1}/{len(splits)}: "
            f"PyTorch MAE={fold_metrics[-1]['mae']:.2f}h | "
            f"sklearn MAE={sklearn_metrics[-1]['mae']:.2f}h"
        )

    summary = summarize_folds(fold_metrics)
    summary.to_csv(out_dir / "cv_summary.csv")
    summarize_folds(sklearn_metrics).to_csv(out_dir / "cv_summary_sklearn_ridge.csv")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--study_name", type=str, default="linear")
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_training_dataset()
    X, y = split_features_target(df)
    splits = make_cv_splits(df)
    print(f"  rows={len(df)} | folds={len(splits)}\n")

    out_dir = RESULTS_DIR / "linear"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Optuna study -------------------------------------------------------
    print(f"Running Optuna study with {args.n_trials} trials...")
    objective = make_objective(
        factory=LinearRegressor,
        suggest_fn=suggest_linear,
        X=X, y=y, splits=splits,
        extra_suggest=_suggest_extras,
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

    trials_df = trials_to_dataframe(study)
    trials_df.to_csv(out_dir / "trials.csv", index=False)

    with open(out_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)

    # --- Re-evaluate the best config ---------------------------------------
    print("Re-evaluating best configuration...")
    best_summary = _evaluate_best(dict(study.best_params), X, y, splits, out_dir)
    print("\n=== Linear model — best-config CV summary ===")
    print(best_summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    main()

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

# Treat type hints as plain text (lets you write `int | None` everywhere safely).
from __future__ import annotations

import argparse  # parses the command-line --flags below
import json
import os
import sys
import warnings
from pathlib import Path  # object-oriented file paths

import numpy as np
import pandas as pd

# Quiet pytabkit / Lightning chatter.
# Hide non-critical UserWarnings so the console output stays readable.
warnings.filterwarnings("ignore", category=UserWarning)
# `os.environ` is the dict of environment variables. `.setdefault` only sets the
# variable if it isn't already set, so a value chosen by the user is respected.
os.environ.setdefault("PYTORCH_LIGHTNING_DISABLE_LITLOGGER", "1")

# Put the package's src/ folder first on the import path so `import ports_dfl...`
# works when running this script directly. `.parents[1]` is two folders up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ports_dfl.config import OPTUNA_DB_DIR, RESULTS_DIR, SEED
from ports_dfl.data.encoders import build_preprocessor
from ports_dfl.data.loader import load_training_dataset, split_features_target
from ports_dfl.data.splits import make_cv_splits
from ports_dfl.metrics.regression import all_metrics, summarize_folds
from ports_dfl.models.realmlp import RealMLP
from ports_dfl.tuning.runner import make_objective, run_study, trials_to_dataframe
from ports_dfl.tuning.search_spaces import suggest_realmlp


# Leading-underscore name = internal helper. `name: str` / `out_dir: Path` are
# type hints; `-> pd.DataFrame` says it returns a pandas table.
def _evaluate(name: str, model_kwargs: dict, X, y, splits, out_dir: Path) -> pd.DataFrame:
    """CV-evaluate a single hyperparameter configuration."""
    # `: list[dict]` annotates the empty list as "a list of dicts" (one per fold).
    fold_metrics: list[dict] = []
    # enumerate gives (fold_idx, split); each split is unpacked into train/val indices.
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # `.iloc[idx]` selects rows by integer position.
        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        # `.to_numpy()` converts the pandas column to a plain numpy array.
        y_train = y.iloc[train_idx].to_numpy()
        y_val = y.iloc[val_idx].to_numpy()

        # Fresh preprocessor each fold: fit on train only, then apply to val,
        # so the validation rows never leak into the encoding/scaling.
        pre = build_preprocessor(categorical_strategy="target")
        # `.astype(np.float32)` casts to 32-bit floats (expected by the model).
        X_train = pre.fit_transform(X_train_raw, y_train).astype(np.float32)
        X_val = pre.transform(X_val_raw).astype(np.float32)

        # `**model_kwargs` unpacks the dict into keyword args. `.shape[1]` is the
        # number of feature columns after preprocessing.
        model = RealMLP(input_dim=X_train.shape[1], **model_kwargs)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        fold_metrics.append(all_metrics(y_val, preds))
        # f-string with {curly braces}; `[-1]` is this fold's just-added metrics,
        # `:.2f` formats to 2 decimals.
        print(
            f"  [{name}] fold {fold_idx + 1}/{len(splits)}: "
            f"MAE={fold_metrics[-1]['mae']:.2f}h"
        )
    # Average per-fold metrics into one row and save to a name-tagged CSV.
    summary = summarize_folds(fold_metrics)
    summary.to_csv(out_dir / f"cv_summary_{name}.csv")
    return summary


# `-> None`: returns nothing. main() holds the whole pipeline.
def main() -> None:
    # Define the command-line flags. `type=int` converts the text to an int;
    # `default=...` is used when the flag is omitted. parse_args() returns an
    # object whose attributes are the flag names (args.n_trials, args.n_epochs...).
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--n_epochs", type=int, default=128)
    parser.add_argument("--study_name", type=str, default="realmlp")
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_training_dataset()
    X, y = split_features_target(df)  # features X, target y
    splits = make_cv_splits(df)  # list of (train_idx, val_idx) for 5-fold CV
    print(f"  rows={len(df)} | folds={len(splits)}\n")

    out_dir = RESULTS_DIR / "realmlp"
    # Make the output folder; parents=True creates missing parents, exist_ok=True
    # avoids erroring if it already exists.
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Stock RealMLP defaults -------------------------------------------
    # Baseline run: RealMLP with its built-in defaults (only n_epochs overridden),
    # no tuning. Gives a reference score to compare the tuned model against.
    print("Evaluating RealMLP with stock TD defaults...")
    stock_summary = _evaluate(
        "stock", {"n_epochs": args.n_epochs}, X, y, splits, out_dir
    )
    print("Stock CV summary:")
    # `.to_string(...)` renders the table as text; the `lambda v: f"{v:.3f}"` is a
    # tiny inline function that formats each number to 3 decimals.
    print(stock_summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print()

    # --- Tuned RealMLP -----------------------------------------------------
    print(f"Running Optuna study with {args.n_trials} trials...")

    # Wrap the model factory so n_epochs propagates from CLI.
    # This nested function "closes over" args.n_epochs (a closure), forcing every
    # tuned model Optuna builds to use the CLI epoch count. `**hp` are the
    # hyperparameters Optuna suggests for each trial.
    def factory(input_dim: int, **hp) -> RealMLP:
        return RealMLP(input_dim=input_dim, n_epochs=args.n_epochs, **hp)

    # Build the objective Optuna minimizes (suggest params -> CV -> return score).
    objective = make_objective(
        factory=factory,
        suggest_fn=suggest_realmlp,
        X=X, y=y, splits=splits,
    )
    # Run the search; seed=SEED makes the trial sequence reproducible.
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

    # Save all trials to CSV (index=False drops the row-number column) and the
    # winning params to JSON. `with open(...) as f:` auto-closes the file; "w"
    # overwrites; json.dump(..., indent=2) pretty-prints the dict.
    trials_to_dataframe(study).to_csv(out_dir / "trials.csv", index=False)
    with open(out_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)

    print("Re-evaluating best configuration...")
    # `dict(study.best_params)` passes a copy so the helper can't mutate Optuna's
    # original best_params. NOTE: unlike the tuning study above (which forced
    # n_epochs via `factory`), this re-evaluation does NOT pass n_epochs, so the
    # tuned re-run uses RealMLP's default epoch count. See REPORTED findings.
    tuned_summary = _evaluate("tuned", dict(study.best_params), X, y, splits, out_dir)
    print("\n=== Tuned RealMLP — CV summary ===")
    print(tuned_summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nResults written to {out_dir}/")


# Run main() only when executed directly, not when imported as a module.
if __name__ == "__main__":
    main()

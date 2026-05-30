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

# `from __future__ import annotations` makes all type hints be treated as plain
# text (not evaluated at runtime). This lets you write hints like `int | None`
# even on older Python versions and avoids import-order headaches.
from __future__ import annotations

import argparse  # standard library for parsing command-line arguments (the --flags)
import json
import sys
from pathlib import Path  # object-oriented file paths; nicer than raw strings

import numpy as np
import optuna
import pandas as pd
# `import X as Y` gives the import a short local alias. Here sklearn's Ridge is
# renamed to SklearnRidge so it can't be confused with this repo's own Ridge.
from sklearn.linear_model import Ridge as SklearnRidge

# Make the package importable when running the script directly.
# `__file__` is this script's path. `.resolve()` turns it absolute, `.parents[1]`
# goes up two folders (scripts -> prediction_models), and `/ "src"` appends the
# src folder. `sys.path.insert(0, ...)` puts that folder first on Python's import
# search path so `import ports_dfl...` below can find the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ports_dfl.config import OPTUNA_DB_DIR, RESULTS_DIR, SEED
from ports_dfl.data.encoders import build_preprocessor
from ports_dfl.data.loader import load_training_dataset, split_features_target
from ports_dfl.data.splits import make_cv_splits
from ports_dfl.metrics.regression import all_metrics, summarize_folds
from ports_dfl.models.linear import LinearRegressor
from ports_dfl.tuning.runner import make_objective, run_study, trials_to_dataframe
from ports_dfl.tuning.search_spaces import suggest_linear


# A leading underscore in a name (like `_suggest_extras`) is a Python convention
# meaning "internal helper" - not meant to be imported/used from outside this file.
# `trial: optuna.Trial` and `-> dict` are type hints: they document the expected
# argument type and the return type. They are not enforced at runtime.
def _suggest_extras(trial: optuna.Trial) -> dict:
    """Tunable preprocessing choices for the linear model."""
    # Optuna picks one option per trial from each list; over many trials it
    # learns which preprocessing choices give the best score.
    return {
        "categorical_strategy": trial.suggest_categorical(
            "categorical_strategy", ["target", "onehot"]
        ),
        "numeric_scaler": trial.suggest_categorical(
            "numeric_scaler", ["standard", "robust"]
        ),
    }


# `pd.DataFrame`/`pd.Series` hints just say "this argument is a pandas table /
# column". `splits` has no hint (it's a list of fold index pairs).
def _evaluate_best(
    best_params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    splits,
    out_dir: Path,
) -> pd.DataFrame:
    """Re-run the best configuration with full bookkeeping."""
    # `.pop(key, default)` removes the key from the dict and returns its value
    # (or the default if missing). We pull the two preprocessing-only choices out
    # so that what remains in `best_params` is only the model's hyperparameters.
    cat_strategy = best_params.pop("categorical_strategy", "target")
    scaler = best_params.pop("numeric_scaler", "standard")

    fold_metrics = []  # MAE/etc. for the PyTorch Ridge, one dict per fold
    sklearn_metrics = []  # closed-form sanity check
    # `enumerate(splits)` yields (index, item) pairs so we get the fold number too.
    # Each split is a (train_idx, val_idx) pair, unpacked here in the for-loop.
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # `.iloc[idx]` selects rows by integer position using the fold's indices.
        X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
        # `.to_numpy()` converts the pandas column to a plain numpy array.
        y_train = y.iloc[train_idx].to_numpy()
        y_val = y.iloc[val_idx].to_numpy()

        # Build a fresh preprocessor per fold so it only "sees" this fold's
        # training data (prevents leakage from validation rows).
        pre = build_preprocessor(
            categorical_strategy=cat_strategy, numeric_scaler=scaler
        )
        # fit_transform learns + applies on train; transform only applies on val.
        # `.astype(np.float32)` casts to 32-bit floats (what PyTorch expects).
        X_train = pre.fit_transform(X_train_raw, y_train).astype(np.float32)
        X_val = pre.transform(X_val_raw).astype(np.float32)

        # PyTorch Ridge.
        # `**best_params` unpacks the dict into keyword args (e.g. weight_decay=...).
        # `.shape[1]` is the number of columns (features) after preprocessing.
        model = LinearRegressor(input_dim=X_train.shape[1], **best_params)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        fold_metrics.append(all_metrics(y_val, preds))

        # Closed-form Ridge baseline (sanity check). sklearn's alpha is roughly
        # the PyTorch weight_decay times the number of samples; clamp to a tiny
        # positive floor so alpha is never zero.
        ridge_alpha = max(
            best_params.get("weight_decay", 1e-3) * len(X_train), 1e-6
        )
        sk = SklearnRidge(alpha=ridge_alpha, random_state=SEED).fit(X_train, y_train)
        sklearn_metrics.append(all_metrics(y_val, sk.predict(X_val)))

        # f-strings (the `f"..."` prefix) let you drop variables inside {curly
        # braces}; `:.2f` formats a number to 2 decimal places. `[-1]` is the
        # last item just appended (this fold's metrics).
        print(
            f"  fold {fold_idx + 1}/{len(splits)}: "
            f"PyTorch MAE={fold_metrics[-1]['mae']:.2f}h | "
            f"sklearn MAE={sklearn_metrics[-1]['mae']:.2f}h"
        )

    # Average the per-fold metrics into one summary table and save both to CSV.
    summary = summarize_folds(fold_metrics)
    summary.to_csv(out_dir / "cv_summary.csv")
    summarize_folds(sklearn_metrics).to_csv(out_dir / "cv_summary_sklearn_ridge.csv")
    return summary


# `-> None` says this function returns nothing. Putting the work in main() (rather
# than at top level) keeps names local and lets us guard it with the __main__ check.
def main() -> None:
    # argparse reads command-line flags. `add_argument("--n_trials", ...)` defines
    # an optional flag; `type=int` converts the text to an int; `default=50` is used
    # when the flag is omitted. `parse_args()` reads sys.argv and returns an object
    # whose attributes are the flag names (e.g. args.n_trials, args.study_name).
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--study_name", type=str, default="linear")
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_training_dataset()
    X, y = split_features_target(df)  # split the table into features X and target y
    splits = make_cv_splits(df)  # list of (train_idx, val_idx) for 5-fold CV
    print(f"  rows={len(df)} | folds={len(splits)}\n")

    out_dir = RESULTS_DIR / "linear"
    # Create the output folder. `parents=True` also makes missing parent folders;
    # `exist_ok=True` means "don't error if it already exists".
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Optuna study -------------------------------------------------------
    print(f"Running Optuna study with {args.n_trials} trials...")
    # Build the function Optuna minimizes: each trial picks hyperparameters, runs
    # CV, and returns a score. `extra_suggest` adds the preprocessing choices above.
    objective = make_objective(
        factory=LinearRegressor,
        suggest_fn=suggest_linear,
        X=X, y=y, splits=splits,
        extra_suggest=_suggest_extras,
    )
    # Run the search for n_trials iterations; `seed=SEED` makes it reproducible.
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

    # Save every trial's params + score, then the best params as JSON.
    trials_df = trials_to_dataframe(study)
    trials_df.to_csv(out_dir / "trials.csv", index=False)  # index=False: no row numbers

    # `with open(...) as f:` opens the file and guarantees it's closed afterward,
    # even if an error occurs. "w" = write (overwrite). json.dump writes the dict
    # as JSON text; indent=2 pretty-prints it.
    with open(out_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)

    # --- Re-evaluate the best config ---------------------------------------
    print("Re-evaluating best configuration...")
    # `dict(study.best_params)` passes a COPY so _evaluate_best's .pop() calls
    # don't mutate Optuna's original best_params dict.
    best_summary = _evaluate_best(dict(study.best_params), X, y, splits, out_dir)
    print("\n=== Linear model — best-config CV summary ===")
    print(best_summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nResults written to {out_dir}/")


# This block runs only when the file is executed directly (python run_linear.py),
# not when it is imported as a module elsewhere. It's the standard script entry point.
if __name__ == "__main__":
    main()

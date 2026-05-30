"""Tune and CV-evaluate NODE (Neural Oblivious Decision Ensembles).

Usage:
    python scripts/run_node.py [--n_trials 30]
"""

# `from __future__ import annotations` lets us write modern type hints (like
# `dict` or `X | None`) even on older Python versions. It must be the first import.
from __future__ import annotations

import argparse  # builds the command-line interface (the --flags below)
import json  # read/write JSON files (used to save the best hyperparameters)
import sys  # access to the interpreter; here used to tweak the import search path
import warnings
from pathlib import Path  # object-oriented file/folder paths

import numpy as np
import pandas as pd

# Hide noisy UserWarnings (e.g. from sklearn/torch) so the console output stays readable.
warnings.filterwarnings("ignore", category=UserWarning)
# Add the repo's `src/` folder to Python's import search path so the
# `ports_dfl` package can be imported even when this script is run directly.
# `__file__` = this script's path; `.resolve()` makes it absolute;
# `.parents[1]` goes up two folders (scripts/ -> prediction_models/), then `/ "src"`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ports_dfl.config import OPTUNA_DB_DIR, RESULTS_DIR, SEED
from ports_dfl.data.encoders import build_preprocessor
from ports_dfl.data.loader import load_training_dataset, split_features_target
from ports_dfl.data.splits import make_cv_splits
from ports_dfl.metrics.regression import all_metrics, summarize_folds
from ports_dfl.models.node import NODE
from ports_dfl.tuning.runner import make_objective, run_study, trials_to_dataframe
from ports_dfl.tuning.search_spaces import suggest_node


# Leading `_` marks this as a "private" helper (convention only, not enforced).
# `best_params: dict` etc. are type hints; `-> pd.DataFrame` is the return type.
def _evaluate_best(best_params: dict, X, y, splits, max_epochs: int, out_dir: Path) -> pd.DataFrame:
    """Re-run best configuration with full bookkeeping."""
    fold_metrics = []  # will collect one metrics-dict per cross-validation fold
    # `enumerate` pairs each item with its index (0, 1, 2, ...). The `(train_idx,
    # val_idx)` part unpacks each split tuple into its two index arrays.
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # `.iloc[...]` selects rows by integer position; grab this fold's train/val rows.
        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        # `.to_numpy()` turns the pandas target Series into a plain numpy array.
        y_train = y.iloc[train_idx].to_numpy()
        y_val = y.iloc[val_idx].to_numpy()

        # Build a FRESH preprocessor per fold and fit it only on this fold's training
        # data — this prevents validation info leaking into the transform.
        pre = build_preprocessor(categorical_strategy="target")
        # `fit_transform` learns encodings/scaling from train data then applies them;
        # `.astype(np.float32)` casts to 32-bit floats (smaller/faster for NN models).
        X_train = pre.fit_transform(X_train_raw, y_train).astype(np.float32)
        # `transform` (no fit) applies the already-learned encodings to val data.
        X_val = pre.transform(X_val_raw).astype(np.float32)

        # `**best_params` "splats" the dict into keyword arguments (e.g. lr=0.01).
        # `X_train.shape[1]` is the number of feature columns after preprocessing.
        model = NODE(input_dim=X_train.shape[1], max_epochs=max_epochs, **best_params)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        fold_metrics.append(all_metrics(y_val, preds))
        # f-string: `{...}` embeds values; `:.2f` formats a float to 2 decimals.
        # `fold_metrics[-1]` is the dict we just appended (-1 = last item).
        print(f"  fold {fold_idx + 1}/{len(splits)}: MAE={fold_metrics[-1]['mae']:.2f}h")
    # Collapse per-fold metrics into a table with mean/std summary rows.
    summary = summarize_folds(fold_metrics)
    # `.to_csv(path)` writes the DataFrame to disk; `out_dir / "..."` joins the path.
    summary.to_csv(out_dir / "cv_summary.csv")
    return summary


# `def main() -> None:` defines the entry-point function; `-> None` = returns nothing.
def main() -> None:
    # ArgumentParser reads command-line options. Each `add_argument` defines one
    # `--flag`; `type=int` converts the text to an integer; `default=...` is used
    # when the flag is omitted. So `--n_trials 50` would set args.n_trials = 50.
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--max_epochs", type=int, default=128)
    parser.add_argument("--study_name", type=str, default="node")
    # Actually parse what the user typed; results are read as attributes (args.n_trials).
    args = parser.parse_args()

    print("Loading dataset...")
    df = load_training_dataset()
    # Split the table into feature columns (X) and the target column (y).
    X, y = split_features_target(df)
    # Precompute the cross-validation folds (list of (train_idx, val_idx) pairs).
    splits = make_cv_splits(df)
    print(f"  rows={len(df)} | folds={len(splits)}\n")

    # Output folder for this model's results; create it (and parents) if missing.
    out_dir = RESULTS_DIR / "node"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Optuna study with {args.n_trials} trials...")

    # A nested helper that builds a fresh NODE from hyperparameters. `**hp` collects
    # any extra keyword args into a dict. Defining it here lets it "capture" args.max_epochs.
    def factory(input_dim: int, **hp) -> NODE:
        return NODE(input_dim=input_dim, max_epochs=args.max_epochs, **hp)

    # Wire the factory, search space, and data into a single objective(trial) function
    # that Optuna will call once per trial (it returns the mean cross-validation MAE).
    objective = make_objective(
        factory=factory,
        suggest_fn=suggest_node,
        X=X, y=y, splits=splits,
    )
    # Run the hyperparameter search. Results persist to a SQLite DB so runs are resumable;
    # passing SEED makes the search reproducible.
    study = run_study(
        study_name=args.study_name,
        objective=objective,
        n_trials=args.n_trials,
        storage_dir=OPTUNA_DB_DIR,
        seed=SEED,
    )

    # `\n` is a newline. `study.best_*` holds the best trial found by the search.
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best mean-fold MAE: {study.best_value:.3f}h")
    print(f"Best params: {study.best_params}\n")

    # Save every trial to CSV; `index=False` omits the DataFrame's row numbers.
    trials_to_dataframe(study).to_csv(out_dir / "trials.csv", index=False)
    # `with open(...) as f:` opens the file and guarantees it is closed afterward
    # (even on error). "w" = write/overwrite; `encoding="utf-8"` makes the text portable.
    with open(out_dir / "best_config.json", "w", encoding="utf-8") as f:
        # Write the best hyperparameters as pretty-printed JSON (indent=2 spaces).
        json.dump(study.best_params, f, indent=2)

    print("Re-evaluating best configuration...")
    # `dict(study.best_params)` makes a fresh copy so `_evaluate_best` can pop/mutate
    # it without corrupting the study's stored params.
    summary = _evaluate_best(dict(study.best_params), X, y, splits, args.max_epochs, out_dir)
    print("\n=== NODE — best-config CV summary ===")
    # `to_string` renders the whole DataFrame as text; `float_format` is a lambda
    # (a tiny inline function) applied to each float to show 3 decimals.
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nResults written to {out_dir}/")


# This block runs only when the file is executed directly (e.g. `python run_node.py`),
# NOT when it is imported by another module. `__name__` is the special string "__main__"
# in the script being run directly.
if __name__ == "__main__":
    main()

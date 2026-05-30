"""Run sanity-floor baselines: global mean and group mean per Sitio / Servicio / Tipo nave.

Outputs ``results/baselines/cv_summary.csv`` with mean ± std MAE / RMSE / R² / MAPE
per baseline. Any real model must beat these by a noticeable margin.

Usage:
    python scripts/run_baselines.py
"""

# `from __future__ import annotations` lets us write modern type hints (like `str | None`)
# even on older Python versions; hints become plain strings and are not evaluated at runtime.
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# `sys.path` is the list of folders Python searches for importable code. Inserting our `src`
# folder at the front (index 0) lets `import ports_dfl...` work when this script is run directly.
# `Path(__file__)` is this file's path; `.resolve()` makes it absolute; `.parents[1]` goes up
# two folders (parents[0] = scripts/, parents[1] = prediction_models/), then we append "src".
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ports_dfl.config import RESULTS_DIR, TARGET_COL
from ports_dfl.data.loader import load_training_dataset, split_features_target
from ports_dfl.data.splits import make_cv_splits
from ports_dfl.metrics.regression import all_metrics, summarize_folds
from ports_dfl.models.baselines import GlobalMeanBaseline, GroupMeanBaseline


# Type hints after each `:` document the expected argument types (for humans and editors); they
# do not enforce anything at runtime. `factory` and `splits` are left unannotated here.
# `-> tuple[str, pd.DataFrame]` says this function returns a (name, table) pair.
def evaluate_baseline(
    name: str,
    factory,
    X: pd.DataFrame,
    y: pd.Series,
    splits,
) -> tuple[str, pd.DataFrame]:
    """Run a single baseline through CV, returning a per-fold summary table."""
    fold_metrics = []  # one metrics dict per cross-validation fold
    # `enumerate` yields (counter, item) so we get a fold number alongside each split.
    # Each split is a (train_idx, val_idx) pair, unpacked directly in the loop header.
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        model = factory()  # build a fresh, untrained model for this fold (avoids leaking state)
        # `.iloc[train_idx]` selects rows by integer position using the fold's index array.
        model.fit(X.iloc[train_idx], y.iloc[train_idx])  # learn from the training rows
        preds = model.predict(X.iloc[val_idx])  # predict on the held-out validation rows
        # `.to_numpy()` converts the pandas Series to a plain numpy array for the metric functions.
        fold_metrics.append(all_metrics(y.iloc[val_idx].to_numpy(), preds))
        # f-strings (the `f"..."` prefix) insert variables inside `{}`; `:.2f` formats a float
        # to 2 decimal places. `fold_metrics[-1]` is the dict we just appended (last item).
        print(
            f"  [{name}] fold {fold_idx + 1}/{len(splits)} "
            f"MAE={fold_metrics[-1]['mae']:.2f}h"
        )
    summary = summarize_folds(fold_metrics)  # aggregate folds into mean/std rows
    return name, summary


# `def main() -> None:` defines the entry point; `-> None` means it returns nothing useful.
def main() -> None:
    print("Loading training dataset...")
    df = load_training_dataset()  # load the full table (rows = port calls, columns = features)
    X, y = split_features_target(df)  # split into inputs (X) and the target column (y)
    splits = make_cv_splits(df)  # build the list of (train_idx, val_idx) cross-validation folds
    print(f"Loaded {len(df)} rows; running {len(splits)}-fold CV.\n")

    # List of (name, factory) pairs. Each `lambda` is a tiny no-argument function that, when
    # called, builds a NEW baseline model. We pass the factory (not a built model) so every fold
    # gets its own fresh instance instead of reusing one trained model across folds.
    baselines = [
        ("global_mean", lambda: GlobalMeanBaseline()),
        ("group_mean__sitio", lambda: GroupMeanBaseline("Sitio")),
        ("group_mean__servicio", lambda: GroupMeanBaseline("Servicio")),
        ("group_mean__tipo_nave", lambda: GroupMeanBaseline("Tipo nave (agrupado)")),
    ]

    # `/` joins paths the OS-correct way. `parents=True` also creates missing parent folders;
    # `exist_ok=True` means "do not error if the folder already exists".
    out_dir = RESULTS_DIR / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []  # collect one summary dict per baseline; type hint documents the shape
    for name, factory in baselines:
        print(f"Evaluating {name}...")
        # `_` is the conventional name for a value we intentionally ignore (here, the returned name).
        _, summary = evaluate_baseline(name, factory, X, y, splits)
        # `.to_csv(path)` writes the DataFrame to a CSV file (index/row labels included by default).
        summary.to_csv(out_dir / f"{name}.csv")
        # `.loc["mean"]` selects the row labeled "mean"; `.to_dict()` turns it into a Python dict.
        mean_row = summary.loc["mean"].to_dict()
        std_row = summary.loc["std"].to_dict()
        rows.append(
            {
                "baseline": name,
                "mae_mean": mean_row["mae"],
                "mae_std": std_row["mae"],
                "rmse_mean": mean_row["rmse"],
                "rmse_std": std_row["rmse"],
                "r2_mean": mean_row["r2"],
                "r2_std": std_row["r2"],
                "mape_mean": mean_row["mape"],
                "mape_std": std_row["mape"],
            }
        )
        print()

    # `pd.DataFrame(rows)` turns the list of dicts into a table (one row per dict, keys = columns).
    # `.sort_values("mae_mean")` orders best-first (lowest error); `.reset_index(drop=True)`
    # renumbers the rows 0..N and discards the old (now scrambled) index.
    summary_df = pd.DataFrame(rows).sort_values("mae_mean").reset_index(drop=True)
    summary_path = out_dir / "cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)  # index=False: don't write the 0..N row numbers

    print("=== Baseline summary (sorted by MAE) ===")
    # `.to_string` renders the whole table as text; `float_format` applies a function to each
    # float cell to round it to 3 decimals for readable console output.
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWritten to {summary_path}")
    # `.mean()` / `.std()` summarize the target column so you can sanity-check the baseline errors.
    print(f"Target ({TARGET_COL}) global mean = {y.mean():.2f}h, std = {y.std():.2f}h")


# This block only runs when the file is executed directly (e.g. `python run_baselines.py`),
# not when it is imported as a module elsewhere. It's the standard Python script entry point.
if __name__ == "__main__":
    main()

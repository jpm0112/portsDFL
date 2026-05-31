"""Run sanity-floor baselines: global mean and group mean per Sitio / Servicio / Tipo nave.

Outputs ``results/baselines/cv_summary.csv`` with mean ± std MAE / RMSE / R² / MAPE
per baseline. Any real model must beat these by a noticeable margin.

Usage:
    python scripts/run_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make the package importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ports_dfl.config import RESULTS_DIR, TARGET_COL
from ports_dfl.data.loader import load_training_dataset, split_features_target
from ports_dfl.data.splits import make_cv_splits
from ports_dfl.metrics.regression import all_metrics, summarize_folds
from ports_dfl.models.baselines import GlobalMeanBaseline, GroupMeanBaseline


def evaluate_baseline(
    name: str,
    factory,
    X: pd.DataFrame,
    y: pd.Series,
    splits,
) -> tuple[str, pd.DataFrame]:
    """Run a single baseline through CV, returning a per-fold summary table."""
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        model = factory()  # fresh model per fold avoids leaking state across folds
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        fold_metrics.append(all_metrics(y.iloc[val_idx].to_numpy(), preds))
        print(
            f"  [{name}] fold {fold_idx + 1}/{len(splits)} "
            f"MAE={fold_metrics[-1]['mae']:.2f}h"
        )
    summary = summarize_folds(fold_metrics)
    return name, summary


def main() -> None:
    print("Loading training dataset...")
    df = load_training_dataset()
    X, y = split_features_target(df)
    splits = make_cv_splits(df)
    print(f"Loaded {len(df)} rows; running {len(splits)}-fold CV.\n")

    # Pass factories (not built models) so every fold gets its own fresh instance.
    baselines = [
        ("global_mean", lambda: GlobalMeanBaseline()),
        ("group_mean__sitio", lambda: GroupMeanBaseline("Sitio")),
        ("group_mean__servicio", lambda: GroupMeanBaseline("Servicio")),
        ("group_mean__tipo_nave", lambda: GroupMeanBaseline("Tipo nave (agrupado)")),
    ]

    out_dir = RESULTS_DIR / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for name, factory in baselines:
        print(f"Evaluating {name}...")
        _, summary = evaluate_baseline(name, factory, X, y, splits)
        summary.to_csv(out_dir / f"{name}.csv")
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

    summary_df = pd.DataFrame(rows).sort_values("mae_mean").reset_index(drop=True)
    summary_path = out_dir / "cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("=== Baseline summary (sorted by MAE) ===")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWritten to {summary_path}")
    print(f"Target ({TARGET_COL}) global mean = {y.mean():.2f}h, std = {y.std():.2f}h")


if __name__ == "__main__":
    main()

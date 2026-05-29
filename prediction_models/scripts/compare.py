"""Aggregate per-model results into a single comparison table.

Reads CV summary CSVs from results/{baselines,linear,realmlp,tabm,node}/
and prints a unified ranking table. Also pulls the real-DBAP demo outcome
(predictive + decision-quality metrics) if present.

Usage:
    python scripts/compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ports_dfl.config import RESULTS_DIR


_SUMMARY_CANDIDATES = ("cv_summary.csv", "cv_summary_tuned.csv", "cv_summary_stock.csv")


def _read_summary(model: str) -> dict | None:
    """Read mean/std rows from a model's first available cv_summary file."""
    for fname in _SUMMARY_CANDIDATES:
        path = RESULTS_DIR / model / fname
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            if "mean" in df.index and "std" in df.index:
                return {
                    "model": f"{model} ({fname.replace('cv_summary', '').strip('_.csv') or 'best'})",
                    "mae_mean": df.loc["mean", "mae"],
                    "mae_std": df.loc["std", "mae"],
                    "rmse_mean": df.loc["mean", "rmse"],
                    "rmse_std": df.loc["std", "rmse"],
                    "r2_mean": df.loc["mean", "r2"],
                    "mape_mean": df.loc["mean", "mape"],
                }
    return None


def _read_baseline_table() -> list[dict]:
    """The baseline summary contains multiple rows in one file."""
    path = RESULTS_DIR / "baselines" / "cv_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "model": f"baseline:{r['baseline']}",
                "mae_mean": r["mae_mean"],
                "mae_std": r["mae_std"],
                "rmse_mean": r["rmse_mean"],
                "rmse_std": r["rmse_std"],
                "r2_mean": r["r2_mean"],
                "mape_mean": r["mape_mean"],
            }
        )
    return rows


def main() -> None:
    rows = _read_baseline_table()
    for model in ["linear", "realmlp", "tabm", "node"]:
        s = _read_summary(model)
        if s is not None:
            rows.append(s)

    if not rows:
        print("No results found. Run scripts/run_*.py first.")
        return

    df = pd.DataFrame(rows).sort_values("mae_mean").reset_index(drop=True)

    print("=" * 70)
    print(" Cross-validated comparison (sorted by MAE) ")
    print("=" * 70)
    pretty_cols = {
        "model": "model",
        "mae_mean": "MAE (h)",
        "mae_std": "± std",
        "rmse_mean": "RMSE (h)",
        "r2_mean": "R²",
        "mape_mean": "MAPE",
    }
    show = df[list(pretty_cols.keys())].rename(columns=pretty_cols)
    print(
        show.to_string(
            index=False,
            float_format=lambda v: f"{v:.3f}",
            justify="left",
        )
    )

    # Real DBAP demo with multi-berth scheduling, if it exists
    real_pred_path = RESULTS_DIR / "dfl_real_bap" / "predictive_summary.csv"
    real_dec_path = RESULTS_DIR / "dfl_real_bap" / "decision_summary.csv"
    if real_pred_path.exists() and real_dec_path.exists():
        print("\n" + "=" * 70)
        print(" Real DBAP (multi-berth scheduling) — predictive ")
        print("=" * 70)
        print(
            pd.read_csv(real_pred_path).to_string(
                index=False, float_format=lambda v: f"{v:.3f}"
            )
        )
        print("\n" + "=" * 70)
        print(" Real DBAP (multi-berth scheduling) — decisions ")
        print("=" * 70)
        # Truncate column names to keep the table readable. "FI" =
        # full-information decision (solved under true τ), the post-hoc
        # optimal benchmark per DFL literature.
        dec = pd.read_csv(real_dec_path)
        rename = {
            "weighted_cost_pred_decision_mean": "cost_pred",
            "weighted_cost_fi_mean": "cost_fi",
            "regret_mean": "regret",
            "regret_relative_pct": "regret_%",
            "makespan_pred_mean": "make_pred",
            "makespan_fi_mean": "make_fi",
            "mean_wait_pred": "wait_pred",
            "mean_wait_fi": "wait_fi",
            "berth_utilization_pred": "util_pred",
            "berth_utilization_fi": "util_fi",
            "fi_assignment_overlap_pct": "fi_assign_overlap_%",
        }
        dec = dec.rename(columns=rename)
        print(dec.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    out = RESULTS_DIR / "comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nWritten unified comparison to {out}")


if __name__ == "__main__":
    main()

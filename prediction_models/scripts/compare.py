"""Aggregate per-model results into a single comparison table.

Reads CV summary CSVs from results/{baselines,linear,realmlp,tabm,node}/
and prints a unified ranking table. Also pulls the real-DBAP demo outcome
(predictive + decision-quality metrics) if present.

Usage:
    python scripts/compare.py
"""

# `from __future__ import annotations` makes all type hints in this file be
# treated as plain text (not evaluated at runtime). This lets us write modern
# hints like `dict | None` even on older Python versions without errors.
from __future__ import annotations

import sys
from pathlib import Path  # Path = object-oriented file paths (nicer than raw strings)

import pandas as pd  # pandas: the table/spreadsheet library; `pd` is the usual nickname

# Make the project's `src/` folder importable. Breakdown:
#   __file__                = path to THIS script
#   Path(__file__).resolve()= turn it into a full absolute path
#   .parents[1]             = go up TWO folders (parents[0] is the immediate parent)
#   / "src"                 = the `/` operator joins paths (Path overrides it)
# sys.path.insert(0, ...) puts that folder at the FRONT of Python's import search
# list, so the `import ports_dfl...` line below can find the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ports_dfl.config import RESULTS_DIR


# A tuple of filenames to try, in priority order. We use whichever one exists
# first for a given model. (A tuple is an immutable list, written with `()`.)
_SUMMARY_CANDIDATES = ("cv_summary.csv", "cv_summary_tuned.csv", "cv_summary_stock.csv")


# `-> dict | None` is a type hint: this function returns either a dict OR None.
# It is documentation only and does not enforce anything at runtime.
def _read_summary(model: str) -> dict | None:
    """Read mean/std rows from a model's first available cv_summary file."""
    # Try each candidate filename; stop at the first one that exists on disk.
    for fname in _SUMMARY_CANDIDATES:
        path = RESULTS_DIR / model / fname
        if path.exists():
            # index_col=0 uses the first CSV column as row labels, so we can
            # later address rows by name (e.g. df.loc["mean", ...]).
            df = pd.read_csv(path, index_col=0)
            # Only use this file if it actually has the summary rows we expect.
            if "mean" in df.index and "std" in df.index:
                # Return a dict of the numbers we care about. The f-string
                # (the f"..." prefix) lets us drop variables into text via {}.
                # NOTE: see REVIEW below — the label-cleaning `.strip('_.csv')`
                # is fragile for the "stock" filename.
                return {
                    "model": f"{model} ({fname.replace('cv_summary', '').strip('_.csv') or 'best'})",
                    # df.loc["mean", "mae"] = value at row "mean", column "mae".
                    "mae_mean": df.loc["mean", "mae"],
                    "mae_std": df.loc["std", "mae"],
                    "rmse_mean": df.loc["mean", "rmse"],
                    "rmse_std": df.loc["std", "rmse"],
                    "r2_mean": df.loc["mean", "r2"],
                    "mape_mean": df.loc["mean", "mape"],
                }
    return None


# Returns a list of dicts (one dict per baseline model). `list[dict]` is the hint.
def _read_baseline_table() -> list[dict]:
    """The baseline summary contains multiple rows in one file."""
    path = RESULTS_DIR / "baselines" / "cv_summary.csv"
    if not path.exists():
        return []  # No file yet -> return an empty list so callers can still loop.
    df = pd.read_csv(path)
    rows = []
    # df.iterrows() yields (row_label, row_data) pairs. We don't need the label,
    # so we name it `_` (a Python convention for "throwaway variable").
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


# `def main() -> None:` declares the main entry point. `-> None` means it
# returns nothing useful; it just does work (printing, writing a file).
def main() -> None:
    # Start with the baseline rows, then add one row per trained model found.
    rows = _read_baseline_table()
    for model in ["linear", "realmlp", "tabm", "node"]:
        s = _read_summary(model)
        if s is not None:  # `is not None` is the correct way to test "has a value"
            rows.append(s)

    # `if not rows:` is True when the list is empty -> nothing to compare.
    if not rows:
        print("No results found. Run scripts/run_*.py first.")
        return  # bail out early

    # Build a DataFrame (a table) from the list of dicts, then:
    #   .sort_values("mae_mean") = sort rows by Mean Absolute Error (lower=better)
    #   .reset_index(drop=True)  = renumber rows 0,1,2,... and discard the old index
    df = pd.DataFrame(rows).sort_values("mae_mean").reset_index(drop=True)

    print("=" * 70)
    print(" Cross-validated comparison (sorted by MAE) ")
    print("=" * 70)
    # Map the internal column names -> human-friendly headers for display.
    pretty_cols = {
        "model": "model",
        "mae_mean": "MAE (h)",
        "mae_std": "± std",
        "rmse_mean": "RMSE (h)",
        "r2_mean": "R²",
        "mape_mean": "MAPE",
    }
    # df[[...]] selects a subset of columns (by a list of names), in this order;
    # .rename(columns=...) then swaps in the pretty headers.
    show = df[list(pretty_cols.keys())].rename(columns=pretty_cols)
    # .to_string() renders the table as text for printing. float_format takes a
    # function applied to each number: `lambda v: f"{v:.3f}"` = format with 3
    # decimals. (A lambda is a tiny inline, unnamed function.)
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
    # Only print this extra section if BOTH files are present.
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
        # Shorten the long column names so the table fits on screen.
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

    # Save the full table to disk. index=False omits the 0,1,2,... row numbers.
    out = RESULTS_DIR / "comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nWritten unified comparison to {out}")


# This block only runs when the file is executed directly (python compare.py),
# NOT when it is imported by another module. It is the standard Python launcher.
if __name__ == "__main__":
    main()

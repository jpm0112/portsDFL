"""
End-to-end evaluation entry point: evaluate -> figures -> report.

Output paths are suffixed by model_key so results from different models
land in distinct files and don't overwrite each other.

Usage (from bayesian_model/ folder):
    python -m src.run_eval --config configs/bhm_baseline.yaml
    python -m src.run_eval --config configs/bhm_m1_covariates.yaml
"""

# Store type hints as text (lets newer hint syntax work everywhere). See note
# in posterior_predictive.py for details.
from __future__ import annotations

import os
# Use 64-bit floats in the JAX backend; setdefault won't override an existing
# value. Set before any pymc-related import (done indirectly via the modules below).
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse  # parses command-line flags like --config
from pathlib import Path

import yaml  # reads the YAML configuration file

# Relative imports from sibling modules in this package: the three pipeline steps.
from .evaluation import run_evaluation     # computes metrics/artifacts from the fitted model
from .figures import make_all_figures      # turns those artifacts into plot files
from .report import write_report           # assembles a Markdown report


# `-> None` means this function returns nothing useful (it runs for side effects).
def main() -> None:
    """Parse args and run the full evaluation + reporting pipeline."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)            # path to the YAML config (mandatory)
    p.add_argument("--n-draws", type=int, default=2000)  # how many posterior draws to evaluate with
    args = p.parse_args()  # parse what the user typed; --n-draws is read as args.n_draws

    config_path = Path(args.config).resolve()  # absolute path to the config file
    base = config_path.parent.parent  # bayesian_model/ (config sits in <base>/configs/<file>.yaml)

    # Open and parse the YAML config into a dict. safe_load avoids executing
    # arbitrary tags in the file.
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    model_key = cfg["model_key"]  # identifies which model (m0..m4); used to name outputs

    # Step 1: run the evaluation; `art` holds the artifacts (metrics, arrays, etc.).
    art = run_evaluation(str(config_path), n_draws=args.n_draws)

    # Step 2: write figures. Per-model figure subdir to keep outputs cleanly partitioned.
    fig_dir = base / "outputs" / "figures" / model_key
    figures = make_all_figures(art, fig_dir)
    print(f"\n[{model_key}] Figures written to: {fig_dir}")
    # `.items()` iterates (key, value) pairs of the returned dict; k is the figure
    # name, p_ is its Path. (Named p_ to avoid clashing with the argparse parser p.)
    for k, p_ in figures.items():
        print(f"  - {k}: {p_.name}")  # p_.name is just the filename, no directory

    # Step 3: write the Markdown report. The diagnostics JSON sits next to the
    # trace file: .with_suffix(".diag.json") swaps the trace's extension for
    # ".diag.json" (e.g. m0.nc -> m0.diag.json).
    diag_path = (base / cfg["output"]["trace"]).with_suffix(".diag.json")
    report_path = base / "outputs" / "reports" / f"{model_key}_evaluation.md"
    write_report(art, figures, diag_path, report_path)
    print(f"[{model_key}] Report written to: {report_path}")


# Run main() only when this file is executed directly, not when imported.
if __name__ == "__main__":
    main()

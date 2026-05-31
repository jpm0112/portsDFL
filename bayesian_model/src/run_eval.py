"""
End-to-end evaluation entry point: evaluate -> figures -> report.

Output paths are suffixed by model_key so results from different models
land in distinct files and don't overwrite each other.

Usage (from bayesian_model/ folder):
    python -m src.run_eval --config configs/bhm_baseline.yaml
    python -m src.run_eval --config configs/bhm_m1_covariates.yaml
"""

from __future__ import annotations

import os
# Set before any pymc-related import (done indirectly via the modules below).
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
from pathlib import Path

import yaml

from .evaluation import run_evaluation
from .figures import make_all_figures
from .report import write_report


def main() -> None:
    """Parse args and run the full evaluation + reporting pipeline."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n-draws", type=int, default=2000)
    args = p.parse_args()

    config_path = Path(args.config).resolve()
    base = config_path.parent.parent  # bayesian_model/ (config sits in <base>/configs/<file>.yaml)

    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    model_key = cfg["model_key"]

    # Step 1: run the evaluation.
    art = run_evaluation(str(config_path), n_draws=args.n_draws)

    # Step 2: write figures (per-model subdir to keep outputs partitioned).
    fig_dir = base / "outputs" / "figures" / model_key
    figures = make_all_figures(art, fig_dir)
    print(f"\n[{model_key}] Figures written to: {fig_dir}")
    for k, p_ in figures.items():
        print(f"  - {k}: {p_.name}")

    # Step 3: write the Markdown report. The diagnostics JSON sits next to the
    # trace file (m0.nc -> m0.diag.json).
    diag_path = (base / cfg["output"]["trace"]).with_suffix(".diag.json")
    report_path = base / "outputs" / "reports" / f"{model_key}_evaluation.md"
    write_report(art, figures, diag_path, report_path)
    print(f"[{model_key}] Report written to: {report_path}")


if __name__ == "__main__":
    main()

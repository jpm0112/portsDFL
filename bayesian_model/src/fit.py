"""
CLI driver to fit any registered Bayesian Hierarchical Model and persist results.

Reads a YAML config, looks up the requested model in the registry, prepares
data (with covariates if needed), builds the model, samples with NUTS,
runs basic diagnostics, and writes the InferenceData trace to disk.

Usage (from the bayesian_model/ directory):
    python -m src.fit --config configs/bhm_baseline.yaml
    python -m src.fit --config configs/bhm_m1_covariates.yaml
"""

from __future__ import annotations

# JAX_ENABLE_X64 must be set before any jax/numpyro import in case
# the user switches sampler.backend to numpyro in the config.
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import json
from pathlib import Path

import arviz as az
import pymc as pm
import yaml

from .data_prep import prepare
from .diagnostics import quick_summary
from .models.registry import MODEL_REGISTRY, build


# Models that need standardized covariates passed to the builder. All others
# get scaler=None and ignore it. Centralized here so the per-model builder
# signatures stay uniform.
MODELS_REQUIRING_COVARIATES = {"m1_covariates"}


def parse_args() -> argparse.Namespace:
    """Parse the --config flag."""
    p = argparse.ArgumentParser(description="Fit a registered BHM for vessel service time.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return p.parse_args()


def load_config(path: str | Path) -> dict:
    """YAML -> dict, UTF-8."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fit_from_config(config_path: str | Path) -> az.InferenceData:
    """
    Run the full fit pipeline for the model named in the config.

    Input:
        config_path: YAML config with at least: model_key, data, priors, sampler, output sections.

    Output:
        arviz.InferenceData. Also written to disk at output.trace.

    Description:
        Resolves all paths relative to the config's parent's parent
        (bayesian_model/) so the same config works regardless of CWD.
    """
    cfg = load_config(config_path)
    model_key = cfg["model_key"]
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model_key '{model_key}'. Registered: {sorted(MODEL_REGISTRY.keys())}")

    base = Path(config_path).resolve().parent.parent  # bayesian_model/
    with_cov = model_key in MODELS_REQUIRING_COVARIATES

    train_df, _test_df, encoding, scaler = prepare(
        csv_path=base / cfg["data"]["path"],
        train_year_max=cfg["data"]["train_year_max"],
        target_col=cfg["data"]["target"],
        vessel_col=cfg["data"]["vessel_col"],
        berth_col=cfg["data"]["berth_col"],
        service_col=cfg["data"]["service_col"],
        with_covariates=True,  # always compute; cheap, and downstream may want them
    )

    model = build(model_key, train_df=train_df, encoding=encoding, scaler=scaler, **cfg["priors"])

    sampler_cfg = cfg["sampler"]
    with model:
        idata = pm.sample(
            draws=sampler_cfg["draws"],
            tune=sampler_cfg["tune"],
            chains=sampler_cfg["chains"],
            target_accept=sampler_cfg["target_accept"],
            random_seed=sampler_cfg["random_seed"],
            nuts_sampler=sampler_cfg["backend"],
            progressbar=False,
            cores=1,  # avoid Windows multiprocess pickling issues
        )

    trace_path = base / cfg["output"]["trace"]
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(idata, str(trace_path))

    summary = quick_summary(idata)
    sidecar = trace_path.with_suffix(".diag.json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{model_key}] Trace written to: {trace_path}")
    print(f"[{model_key}] Diagnostics:      {sidecar}")
    print(json.dumps(summary, indent=2))

    return idata


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    fit_from_config(args.config)


if __name__ == "__main__":
    main()

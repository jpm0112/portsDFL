"""
CLI driver to fit any registered Bayesian Hierarchical Model and persist results.

Reads a YAML config, looks up the requested model in the registry, prepares
data (with covariates if needed), builds the model, samples with NUTS,
runs basic diagnostics, and writes the InferenceData trace to disk.

Usage (from the bayesian_model/ directory):
    python -m src.fit --config configs/bhm_baseline.yaml
    python -m src.fit --config configs/bhm_m1_covariates.yaml
"""

# `from __future__ import annotations` makes all type hints (the `: str`,
# `-> dict` parts below) be treated as plain text instead of being evaluated.
# This lets us write hints like `str | Path` even on older Python versions.
from __future__ import annotations

# JAX_ENABLE_X64 must be set before any jax/numpyro import in case
# the user switches sampler.backend to numpyro in the config.
import os
# os.environ is a dict of environment variables. setdefault sets the key
# only if it is not already set, so we don't override a user-provided value.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse  # builds the command-line interface (the --config flag)
import json      # writes the diagnostics sidecar file
from pathlib import Path  # object-oriented file paths (works on Windows + Linux)

import arviz as az  # ArviZ: post-sampling diagnostics & InferenceData I/O
import pymc as pm   # PyMC: defines the Bayesian model and runs MCMC sampling
import yaml         # reads the human-friendly YAML config file

# Relative imports (the leading dot means "from this same package").
from .data_prep import prepare            # load CSV -> split -> encode -> add log target
from .diagnostics import quick_summary    # compact R-hat / ESS / divergence report
from .models.registry import MODEL_REGISTRY, build  # name -> model-builder lookup


# Models that need standardized covariates passed to the builder. All others
# get scaler=None and ignore it. Centralized here so the per-model builder
# signatures stay uniform.
# `{...}` here is a set literal (not a dict — no key:value pairs). A set is an
# unordered collection of unique items; `x in this_set` is a fast membership test.
MODELS_REQUIRING_COVARIATES = {"m1_covariates"}


# `def name(args) -> ReturnType:` defines a function; the `-> ...` is just a hint
# describing what the function returns (here, argparse's parsed-arguments object).
def parse_args() -> argparse.Namespace:
    """Parse the --config flag."""
    p = argparse.ArgumentParser(description="Fit a registered BHM for vessel service time.")
    # required=True means the program errors out if --config is not given.
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return p.parse_args()


# `str | Path` means the argument may be either a string OR a Path object.
def load_config(path: str | Path) -> dict:
    """YAML -> dict, UTF-8."""
    # A `with` block opens the file and guarantees it is closed afterward,
    # even if an error happens. `as f` names the open file object.
    with open(path, "r", encoding="utf-8") as f:
        # safe_load parses YAML text into Python dicts/lists (the "safe" version
        # refuses to run arbitrary code embedded in the file).
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
    model_key = cfg["model_key"]  # which model to build, e.g. "m0_baseline"
    # Fail fast with a clear message if the config names an unknown model.
    # `f"...{x}..."` is an f-string: text with {expressions} substituted in.
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model_key '{model_key}'. Registered: {sorted(MODEL_REGISTRY.keys())}")

    # .resolve() makes the path absolute; .parent goes up one folder. Two
    # .parent's take us from configs/foo.yaml up to the bayesian_model/ root,
    # so every path in the config is interpreted relative to that root.
    base = Path(config_path).resolve().parent.parent  # bayesian_model/
    with_cov = model_key in MODELS_REQUIRING_COVARIATES

    # Keyword arguments (name=value) make the call self-documenting. The leading
    # underscore in `_test_df` is a Python convention meaning "intentionally
    # unused here" (fit only needs the training rows).
    train_df, _test_df, encoding, scaler = prepare(
        csv_path=base / cfg["data"]["path"],
        train_year_max=cfg["data"]["train_year_max"],
        target_col=cfg["data"]["target"],
        vessel_col=cfg["data"]["vessel_col"],
        berth_col=cfg["data"]["berth_col"],
        service_col=cfg["data"]["service_col"],
        with_covariates=True,  # always compute; cheap, and downstream may want them
    )

    # build() returns a PyMC model object with all priors/likelihood defined.
    # `**cfg["priors"]` unpacks the priors dict into keyword arguments: each
    # key=value in that dict becomes a named argument to the builder. This lets
    # the YAML config tune prior hyperparameters without touching the code.
    model = build(model_key, train_df=train_df, encoding=encoding, scaler=scaler, **cfg["priors"])

    sampler_cfg = cfg["sampler"]
    # `with model:` enters the model as the active context, so the sampling call
    # below knows which random variables / likelihood to draw the posterior for.
    with model:
        # pm.sample runs MCMC (NUTS by default): it draws samples from the
        # posterior distribution of every parameter given the observed data.
        idata = pm.sample(
            draws=sampler_cfg["draws"],            # kept posterior samples per chain
            tune=sampler_cfg["tune"],              # warm-up steps (discarded; tune the sampler)
            chains=sampler_cfg["chains"],          # independent runs (used to compute R-hat)
            target_accept=sampler_cfg["target_accept"],  # higher (e.g. 0.9) = smaller steps, fewer divergences
            random_seed=sampler_cfg["random_seed"],      # reproducibility
            nuts_sampler=sampler_cfg["backend"],   # "pymc" or "numpyro" backend for NUTS
            progressbar=False,
            cores=1,  # avoid Windows multiprocess pickling issues
        )

    # The `/` operator on Path objects joins paths (base / "traces/m0.nc").
    trace_path = base / cfg["output"]["trace"]
    # Create the output folder (and any missing parents); exist_ok avoids an
    # error if it already exists.
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist the full InferenceData (posterior, sample_stats, ...) to a netCDF
    # file so later scripts can reload it without re-running the sampler.
    az.to_netcdf(idata, str(trace_path))

    # Compute the compact diagnostics dict (worst R-hat, min ESS, divergences).
    summary = quick_summary(idata)
    # .with_suffix swaps the file extension, giving e.g. m0.diag.json next to m0.nc.
    sidecar = trace_path.with_suffix(".diag.json")
    with open(sidecar, "w", encoding="utf-8") as f:
        # json.dump writes the dict to the file; indent=2 pretty-prints it.
        json.dump(summary, f, indent=2)

    print(f"\n[{model_key}] Trace written to: {trace_path}")
    print(f"[{model_key}] Diagnostics:      {sidecar}")
    print(json.dumps(summary, indent=2))

    return idata


# `-> None` means this function returns nothing useful (it just runs side effects).
def main() -> None:
    """CLI entry point."""
    args = parse_args()
    fit_from_config(args.config)


# This block runs only when the file is executed directly (e.g.
# `python -m src.fit ...`), not when it is imported by another module. It is the
# standard Python way to provide a script entry point.
if __name__ == "__main__":
    main()

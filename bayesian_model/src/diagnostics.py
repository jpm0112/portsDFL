"""
Diagnostic utilities for the BHM trace.

quick_summary returns a compact JSON-friendly dict suitable for sidecar
files and CI checks. Heavier diagnostics (full ArviZ summary, posterior
predictive checks, LOO) are exposed as separate functions used from
notebooks or downstream scripts.
"""

from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np


# Conventional thresholds used in the literature (Vehtari et al., 2021;
# Stan reference manual). Crossing them does not necessarily mean the fit
# is broken, but it warrants investigation before trusting the posterior.
# R-hat compares within-chain vs between-chain variance; ~1.0 = chains agree.
RHAT_THRESHOLD = 1.01
# ESS = effective sample size: independent-draw equivalent of an autocorrelated
# chain. Too few means noisy posterior estimates.
ESS_THRESHOLD = 400


def quick_summary(idata: az.InferenceData) -> dict[str, Any]:
    """
    Compute a compact diagnostic summary of a posterior trace.

    Input:
        idata: arviz.InferenceData returned by pm.sample.

    Output:
        Dict with worst-case R-hat, minimum bulk/tail ESS, divergence count,
        and a list of any parameters that exceed the convention thresholds.
        All values are plain Python types so the dict is JSON-serializable.

    Description:
        Used to write a small sidecar JSON next to the netCDF trace and to
        gate test runs / CI. Operates only on the posterior and sample_stats
        groups so it is cheap to call repeatedly.
    """
    # kind="diagnostics" returns only the convergence columns (r_hat, ess_bulk, ess_tail).
    summary = az.summary(idata, kind="diagnostics")

    # Worst (largest) r-hat across all parameters.
    rhat_max = float(summary["r_hat"].max())
    # Smallest ESS across parameters (bulk = center of dist, tail = the extremes).
    ess_bulk_min = float(summary["ess_bulk"].min())
    ess_tail_min = float(summary["ess_tail"].min())

    bad_rhat = summary.index[summary["r_hat"] > RHAT_THRESHOLD].tolist()
    bad_ess = summary.index[summary["ess_bulk"] < ESS_THRESHOLD].tolist()

    # Divergences = steps where the sampler's trajectory blew up; flag regions
    # the sampler couldn't explore reliably.
    n_div = int(idata.sample_stats["diverging"].sum().item())

    return {
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_bulk_min,
        "ess_tail_min": ess_tail_min,
        "n_divergences": n_div,
        "params_high_rhat": bad_rhat,
        "params_low_ess": bad_ess,
        "thresholds": {"rhat_max": RHAT_THRESHOLD, "ess_min": ESS_THRESHOLD},
    }


def full_summary(idata: az.InferenceData) -> "az.utils.InferenceData":
    """
    Return the complete ArviZ summary table (means, SDs, HDIs, R-hat, ESS).

    Input:  idata as above.
    Output: pandas DataFrame from arviz.summary.
    Description: convenience pass-through used in notebooks; not used by fit.py.
    """
    # No kind= argument -> ArviZ returns the full table (means, SDs, HDIs, r_hat/ESS).
    return az.summary(idata)


def posterior_predictive_check(
    idata: az.InferenceData,
    n_draws: int = 200,
) -> np.ndarray:
    """
    Sample replicated log(svc) from the posterior predictive for each obs.

    Input:
        idata: trace that contains a posterior_predictive group (must be
               populated by the caller via pm.sample_posterior_predictive).
        n_draws: number of replicated draws to subsample for plotting.

    Output:
        ndarray of shape (n_draws, n_obs) with replicated log(svc) values.

    Description:
        Designed to be plotted against the observed log(svc) (e.g., overlay
        KDEs) to check that the model captures the marginal distribution.
        Returns raw arrays rather than figures so callers can choose plotting.
    """
    # Guard: caller must generate the replicated data first.
    if "posterior_predictive" not in idata.groups():
        raise ValueError(
            "idata has no posterior_predictive group. "
            "Run pm.sample_posterior_predictive first."
        )
    # Flatten chain+draw into one "sample" axis (see _bhm_predictive_samples).
    pp = idata.posterior_predictive["log_y_obs"].stack(sample=("chain", "draw"))
    rng = np.random.default_rng(0)
    # FIX: clamp to the number of available samples so short traces
    # (chains*draws < n_draws) don't crash rng.choice(replace=False).
    n_take = min(int(n_draws), int(pp.sizes["sample"]))
    sample_idx = rng.choice(pp.sizes["sample"], size=n_take, replace=False)
    return pp.isel(sample=sample_idx).values.T  # (n_take, n_obs)


def compare_loo(traces: dict[str, az.InferenceData]) -> "az.compare":
    """
    Pairwise model comparison via PSIS-LOO expected log predictive density.

    Input:
        traces: dict of model name -> InferenceData with a log_likelihood group.

    Output:
        ArviZ comparison DataFrame ranked by ELPD with standard errors.

    Description:
        Forward to az.compare; provided here so model-comparison logic stays
        co-located with the rest of the diagnostics. Caller is responsible
        for ensuring each trace has log_likelihood (passed idata_kwargs to
        pm.sample with idata_kwargs={'log_likelihood': True}, or via
        pm.compute_log_likelihood after sampling).
    """
    # ic="loo" ranks by PSIS-LOO ELPD (higher = better out-of-sample accuracy),
    # estimated from each trace's log_likelihood group.
    return az.compare(traces, ic="loo")

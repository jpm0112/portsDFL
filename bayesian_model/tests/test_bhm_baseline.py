"""
Synthetic-data parameter recovery test for the M0 BHM.

Procedure:
    1. Pick known true values for alpha0, tau_vessel, tau_berth, tau_service, sigma.
    2. Sample group offsets and observations from the generative model.
    3. Fit M0 with build_model + pm.sample.
    4. Assert posterior 94% HDI contains the truth for each tau and alpha0.
    5. Assert sigma posterior mean is within tight tolerance (well identified).
    6. Assert NUTS produced no divergences.

Why HDI containment rather than mean tolerance: tau parameters in
hierarchical models with a small number of groups are poorly identified
in the mean (the prior dominates) but well-calibrated in the posterior
distribution. Checking that the truth falls inside a high-density interval
is the principled test of model correctness.
"""

# This `from __future__ import` makes every type hint in this file be treated as
# a plain string instead of being evaluated when the file loads. It lets us write
# modern hints (like `X | None`) without breaking on older Python versions.
from __future__ import annotations

import arviz as az      # ArviZ: diagnostics + summaries for Bayesian samples
import numpy as np      # numerical arrays and random-number generation
import pandas as pd     # tabular data (the DataFrame the model consumes)
import pymc as pm       # PyMC: defines the Bayesian model and runs MCMC sampling
import pytest

# `src.*` are the project's own modules. `Encoding` is a small data holder that
# tells the model how many vessel/berth/service groups there are; `build_model`
# constructs the PyMC model we are testing here.
from src.data_prep import Encoding
from src.models.bhm_baseline import build_model


# Known generative parameters.
# These are the "ground truth" values we will bake into fake (synthetic) data.
# Because WE chose them, a correct model fit should recover them. That is the
# whole idea of a parameter-recovery test.
TRUE_TAU_VESSEL = 0.4
TRUE_TAU_BERTH = 0.3
TRUE_TAU_SERVICE = 0.5
TRUE_SIGMA = 0.4
TRUE_ALPHA0 = 3.5

# Group counts large enough to give the posterior decent info on each tau.
# With <6 groups, tau is dominated by the half-normal prior even with lots
# of observations; recovery in the mean becomes unreliable.
N_VESSEL = 8
N_BERTH = 8
N_SERVICE = 12
N_OBS = 2000
RANDOM_SEED = 12345
HDI_PROB = 0.94


# `def name(...) -> ReturnType:` defines a function. The part after `->` is a
# type hint describing what it returns. Here it returns a tuple of two things:
# a pandas DataFrame and an Encoding. The leading underscore in `_generate_synthetic`
# is a convention meaning "internal helper, not part of the public test API".
def _generate_synthetic() -> tuple[pd.DataFrame, Encoding]:
    """
    Sample data from the generative model used by build_model.

    Output:
        (train_df, encoding) where train_df contains vessel_idx, berth_idx,
        service_idx, and log_service_time columns. service_time_hours is
        the exponentiation of log_service_time so the same DataFrame works
        for any consumer that expects either column.

    Description:
        Uses a fixed RNG seed for reproducibility. Indices are uniformly
        sampled, which gives roughly balanced cells; the model should
        recover the true taus regardless of cell balance.
    """
    # Seeded random generator: the same seed always yields the same numbers,
    # so this test is reproducible (it will not randomly pass/fail).
    rng = np.random.default_rng(RANDOM_SEED)
    # Draw one standard-normal value per group. This mirrors the model's
    # "non-centered" parameterization: each group offset = tau * z, where
    # z ~ Normal(0, 1). Building the data the same way the model assumes it
    # is what makes recovery possible.
    z_v = rng.standard_normal(N_VESSEL)
    z_b = rng.standard_normal(N_BERTH)
    z_s = rng.standard_normal(N_SERVICE)
    # Per-group offsets on the log scale (how much each group shifts service time).
    alpha_v = TRUE_TAU_VESSEL * z_v
    alpha_b = TRUE_TAU_BERTH * z_b
    alpha_s = TRUE_TAU_SERVICE * z_s

    # Randomly assign each of the N_OBS observations to a vessel/berth/service
    # group. integers(0, N, size) draws ints in [0, N) (N excluded).
    v_idx = rng.integers(0, N_VESSEL, size=N_OBS)
    b_idx = rng.integers(0, N_BERTH, size=N_OBS)
    s_idx = rng.integers(0, N_SERVICE, size=N_OBS)

    # Expected log service time per observation = global intercept + the offset
    # of each group it belongs to. `alpha_v[v_idx]` is fancy ("vectorized")
    # indexing: for every observation, look up its vessel group's offset.
    mu = TRUE_ALPHA0 + alpha_v[v_idx] + alpha_b[b_idx] + alpha_s[s_idx]
    # Add Normal noise with the true residual sd to get the observed log target.
    log_y = mu + rng.normal(0.0, TRUE_SIGMA, size=N_OBS)

    # Assemble the table the model expects. service_time_hours is the raw
    # (non-log) target: np.exp undoes the log so both column conventions exist.
    df = pd.DataFrame(
        {
            "vessel_idx": v_idx,
            "berth_idx": b_idx,
            "service_idx": s_idx,
            "log_service_time": log_y,
            "service_time_hours": np.exp(log_y),
        }
    )

    # Build an Encoding describing the group counts. The `{f"V{i}": i for i in ...}`
    # is a dict comprehension: it builds a dict mapping each group name (an
    # f-string like "V0", "V1", ...) to its integer index. f-strings let you
    # embed `{i}` directly inside the string text.
    encoding = Encoding(
        vessel={f"V{i}": i for i in range(N_VESSEL)},
        berth={f"B{i}": i for i in range(N_BERTH)},
        service={f"S{i}": i for i in range(N_SERVICE)},
        n_vessel=N_VESSEL,
        n_berth=N_BERTH,
        n_service=N_SERVICE,
    )
    return df, encoding


def _hdi_contains(idata: "az.InferenceData", var: str, truth: float, prob: float) -> bool:
    """
    Check whether the posterior HDI for `var` covers the true value.

    Input:  idata, parameter name, true value, HDI probability mass.
    Output: bool.
    Description: tiny helper to keep the test body readable.
    """
    # az.hdi computes the Highest-Density Interval: the shortest range that
    # holds `prob` (e.g. 94%) of the posterior probability for this parameter.
    # `[var].values` pulls out the two interval endpoints as a NumPy array
    # [low, high]. The chained comparison `low <= truth <= high` is Python
    # shorthand for "truth is between low and high (inclusive)".
    bounds = az.hdi(idata, var_names=[var], hdi_prob=prob)[var].values
    return bool(bounds[0] <= truth <= bounds[1])


def test_recover_known_parameters():
    """
    Posterior 94% HDI covers the truth for taus + alpha0; sigma mean tight;
    NUTS produces no divergences.
    """
    train_df, encoding = _generate_synthetic()

    # Build the PyMC model object bound to our synthetic data. We pass priors
    # that comfortably cover the true values; build_model wires up the
    # intercept, the group taus, and the Normal likelihood internally.
    model = build_model(
        train_df=train_df,
        encoding=encoding,
        scaler=None,
        alpha0_mean=TRUE_ALPHA0,
        alpha0_sd=1.0,
        tau_halfnormal_sd=1.0,
        sigma_halfnormal_sd=1.0,
    )

    # `with model:` is a context manager (a `with` block). Inside it, PyMC knows
    # which model the following commands refer to, so pm.sample fits THIS model.
    with model:
        # pm.sample runs NUTS (a form of MCMC) to draw samples from the
        # posterior — i.e. the plausible parameter values given the data.
        #   draws=1000   -> kept samples per chain (after warm-up)
        #   tune=1000    -> warm-up steps the sampler discards
        #   chains=2     -> independent runs, used to cross-check convergence
        #   target_accept-> higher = smaller steps, fewer divergences
        # `idata` is an ArviZ InferenceData object holding the results.
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            target_accept=0.95,
            random_seed=RANDOM_SEED,
            progressbar=False,
            # Default backend keeps the test free of jax/numpyro requirement.
        )

    # Divergences are sampler warnings that the geometry was too hard to explore;
    # any divergence means the posterior may be untrustworthy. We require zero.
    # `.sum().item()` totals the boolean "diverging" flags into a single number.
    n_div = int(idata.sample_stats["diverging"].sum().item())
    assert n_div == 0, f"NUTS produced {n_div} divergences"

    # Tau and alpha0: check HDI containment (the right calibration test).
    # If the model is correct, the 94% interval should bracket the truth.
    assert _hdi_contains(idata, "tau_vessel", TRUE_TAU_VESSEL, HDI_PROB)
    assert _hdi_contains(idata, "tau_berth", TRUE_TAU_BERTH, HDI_PROB)
    assert _hdi_contains(idata, "tau_service", TRUE_TAU_SERVICE, HDI_PROB)
    assert _hdi_contains(idata, "alpha0", TRUE_ALPHA0, HDI_PROB)

    # Sigma is tightly identified by N_OBS=2000 points.
    # `idata.posterior["sigma"]` holds every sampled sigma value; .mean()
    # averages them into the posterior-mean estimate. With this much data the
    # estimate should land within 0.03 of the true sigma.
    sigma_mean = float(idata.posterior["sigma"].mean())
    assert abs(sigma_mean - TRUE_SIGMA) < 0.03, f"sigma mean {sigma_mean} far from {TRUE_SIGMA}"

"""
Synthetic-data parameter recovery test for M2 (Student-T likelihood).

Generate observations from a Student-T with known nu, fit M2, assert that
the posterior puts substantial mass near the true nu and recovers taus.
"""

# `from __future__ import annotations` makes type hints (like the `-> tuple[...]`
# return annotations below) lazy strings, so newer syntax works on older Pythons.
from __future__ import annotations

import arviz as az          # ArviZ: post-processing/diagnostics for Bayesian samples
import numpy as np          # numerical arrays and random number generation
import pandas as pd         # DataFrames (table-like data)
import pymc as pm           # PyMC: defines probabilistic models and runs MCMC sampling

from src.data_prep import Encoding              # dataclass holding the category->index maps
from src.models.bhm_heavytail import build_model  # builds the M2 PyMC model under test


# Module-level constants (ALL_CAPS by convention = "do not change at runtime").
# These define the size of the fake dataset and the *true* parameter values we
# will try to recover, so we can check the model gets close to the truth.
N_VESSEL = 6
N_BERTH = 6
N_SERVICE = 10
N_OBS = 1500
RANDOM_SEED = 4242

TRUE_ALPHA0 = 3.5
TRUE_TAU_VESSEL = 0.4
TRUE_TAU_BERTH = 0.3
TRUE_TAU_SERVICE = 0.4
TRUE_SIGMA = 0.35
TRUE_NU = 4.0  # heavy-tailed
HDI_PROB = 0.94


# A leading underscore (`_generate_synthetic`) signals "private helper" — used only
# inside this file. `-> tuple[pd.DataFrame, Encoding]` is a type hint saying the
# function returns a 2-item tuple of those types (documentation only, not enforced).
def _generate_synthetic() -> tuple[pd.DataFrame, Encoding]:
    """Sample data with a Student-T noise distribution; return (df, encoding)."""
    # Seeded random generator => the "random" data is identical every run (reproducible test).
    rng = np.random.default_rng(RANDOM_SEED)
    # True group offsets: each is a draw of N standard normals scaled by the true tau.
    # `standard_normal(N)` returns an array of N samples from Normal(0, 1).
    alpha_v = TRUE_TAU_VESSEL * rng.standard_normal(N_VESSEL)
    alpha_b = TRUE_TAU_BERTH * rng.standard_normal(N_BERTH)
    alpha_s = TRUE_TAU_SERVICE * rng.standard_normal(N_SERVICE)

    # Random group memberships for each of the N_OBS observations.
    # `integers(0, N, size=K)` draws K ints in [0, N) (low inclusive, high exclusive).
    v_idx = rng.integers(0, N_VESSEL, size=N_OBS)
    b_idx = rng.integers(0, N_BERTH, size=N_OBS)
    s_idx = rng.integers(0, N_SERVICE, size=N_OBS)
    # Fancy indexing: alpha_v[v_idx] picks the offset for each row's vessel group.
    # mu is the true mean of log(service time) for every observation.
    mu = TRUE_ALPHA0 + alpha_v[v_idx] + alpha_b[b_idx] + alpha_s[s_idx]
    # Add heavy-tailed noise: standard_t draws from a Student-T with TRUE_NU
    # degrees of freedom (small nu => heavier tails / more outliers), scaled by sigma.
    # This is the signal the model must detect to "win" over a plain Normal.
    log_y = mu + TRUE_SIGMA * rng.standard_t(df=TRUE_NU, size=N_OBS)

    # Build the table the model consumes. service_time_hours = exp(log_y) because the
    # model works on the log scale; np.exp undoes the log so both columns are consistent.
    df = pd.DataFrame({
        "vessel_idx": v_idx, "berth_idx": b_idx, "service_idx": s_idx,
        "log_service_time": log_y, "service_time_hours": np.exp(log_y),
    })
    # Encoding maps category names to indices. `{f"V{i}": i for i in range(N)}` is a
    # dict comprehension; f"V{i}" is an f-string that inserts i, e.g. i=2 -> "V2".
    enc = Encoding(
        vessel={f"V{i}": i for i in range(N_VESSEL)},
        berth={f"B{i}": i for i in range(N_BERTH)},
        service={f"S{i}": i for i in range(N_SERVICE)},
        n_vessel=N_VESSEL, n_berth=N_BERTH, n_service=N_SERVICE,
    )
    return df, enc


# pytest auto-discovers and runs any function named test_*. No `self`/class needed.
def test_m2_recovers_heavy_tails():
    """Posterior nu is plausibly heavy-tailed; taus and alpha0 recovered."""
    df, enc = _generate_synthetic()
    # build_model returns a PyMC model wired to this data, with priors deliberately
    # centered on the truth (alpha0_mean=TRUE_ALPHA0) so the test isolates whether
    # sampling can recover the parameters rather than testing prior choices.
    model = build_model(
        df, enc, alpha0_mean=TRUE_ALPHA0, alpha0_sd=1.0,
        tau_halfnormal_sd=1.0, sigma_halfnormal_sd=1.0,
    )
    # `with model:` enters the model as the active context, so pm.sample knows which
    # model to draw from. pm.sample runs MCMC (the NUTS sampler) to approximate the
    # posterior: draws=1000 kept samples per chain after tune=1000 warmup steps,
    # chains=2 independent runs, target_accept=0.95 makes the sampler take smaller,
    # more careful steps (fewer divergences). idata is an ArviZ InferenceData object.
    with model:
        idata = pm.sample(
            draws=1000, tune=1000, chains=2, target_accept=0.95,
            random_seed=RANDOM_SEED, progressbar=False,
        )

    # "Divergences" are NUTS warnings that it could not explore a region reliably;
    # any divergence means the posterior estimate may be biased. .sum() adds them
    # across draws/chains; .item() pulls the single number out of the array.
    n_div = int(idata.sample_stats["diverging"].sum().item())
    assert n_div == 0, f"NUTS produced {n_div} divergences"

    # For each parameter, check its true value lies inside the posterior credible
    # interval. `for var, truth in [...]` unpacks each (name, true-value) pair.
    for var, truth in [
        ("tau_vessel", TRUE_TAU_VESSEL),
        ("tau_berth", TRUE_TAU_BERTH),
        ("tau_service", TRUE_TAU_SERVICE),
        ("alpha0", TRUE_ALPHA0),
    ]:
        # az.hdi computes the Highest Density Interval: the narrowest range holding
        # HDI_PROB (94%) of the posterior mass. `[var].values` extracts the [low, high] array.
        bounds = az.hdi(idata, var_names=[var], hdi_prob=HDI_PROB)[var].values
        # Chained comparison: True only if low <= truth <= high (truth inside the interval).
        assert bounds[0] <= truth <= bounds[1], f"{var} HDI {bounds} misses truth {truth}"

    # nu posterior median should be on the heavy-tail side (< 15).
    # idata.posterior["nu"] holds all posterior draws of nu; np.median summarizes them.
    # A small median nu confirms the model "noticed" the heavy tails we simulated.
    nu_med = float(np.median(idata.posterior["nu"].values))
    assert nu_med < 15, f"nu median {nu_med} suggests model failed to detect heavy tails"

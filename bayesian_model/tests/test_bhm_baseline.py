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

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest

from src.data_prep import Encoding
from src.models.bhm_baseline import build_model


# Known generative parameters: a correct model fit should recover these.
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
    rng = np.random.default_rng(RANDOM_SEED)
    # Build the data the same non-centered way the model assumes (offset =
    # tau * z), which is what makes recovery possible.
    z_v = rng.standard_normal(N_VESSEL)
    z_b = rng.standard_normal(N_BERTH)
    z_s = rng.standard_normal(N_SERVICE)
    alpha_v = TRUE_TAU_VESSEL * z_v
    alpha_b = TRUE_TAU_BERTH * z_b
    alpha_s = TRUE_TAU_SERVICE * z_s

    v_idx = rng.integers(0, N_VESSEL, size=N_OBS)
    b_idx = rng.integers(0, N_BERTH, size=N_OBS)
    s_idx = rng.integers(0, N_SERVICE, size=N_OBS)

    mu = TRUE_ALPHA0 + alpha_v[v_idx] + alpha_b[b_idx] + alpha_s[s_idx]
    log_y = mu + rng.normal(0.0, TRUE_SIGMA, size=N_OBS)

    df = pd.DataFrame(
        {
            "vessel_idx": v_idx,
            "berth_idx": b_idx,
            "service_idx": s_idx,
            "log_service_time": log_y,
            "service_time_hours": np.exp(log_y),
        }
    )

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
    # HDI: the shortest range holding `prob` of the posterior mass.
    bounds = az.hdi(idata, var_names=[var], hdi_prob=prob)[var].values
    return bool(bounds[0] <= truth <= bounds[1])


def test_recover_known_parameters():
    """
    Posterior 94% HDI covers the truth for taus + alpha0; sigma mean tight;
    NUTS produces no divergences.
    """
    train_df, encoding = _generate_synthetic()

    # Priors comfortably cover the true values.
    model = build_model(
        train_df=train_df,
        encoding=encoding,
        scaler=None,
        alpha0_mean=TRUE_ALPHA0,
        alpha0_sd=1.0,
        tau_halfnormal_sd=1.0,
        sigma_halfnormal_sd=1.0,
    )

    with model:
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            target_accept=0.95,
            random_seed=RANDOM_SEED,
            progressbar=False,
            # Default backend keeps the test free of jax/numpyro requirement.
        )

    # Any divergence means the posterior may be untrustworthy; require zero.
    n_div = int(idata.sample_stats["diverging"].sum().item())
    assert n_div == 0, f"NUTS produced {n_div} divergences"

    # HDI containment is the right calibration test for tau/alpha0.
    assert _hdi_contains(idata, "tau_vessel", TRUE_TAU_VESSEL, HDI_PROB)
    assert _hdi_contains(idata, "tau_berth", TRUE_TAU_BERTH, HDI_PROB)
    assert _hdi_contains(idata, "tau_service", TRUE_TAU_SERVICE, HDI_PROB)
    assert _hdi_contains(idata, "alpha0", TRUE_ALPHA0, HDI_PROB)

    # Sigma is tightly identified by N_OBS=2000 points.
    sigma_mean = float(idata.posterior["sigma"].mean())
    assert abs(sigma_mean - TRUE_SIGMA) < 0.03, f"sigma mean {sigma_mean} far from {TRUE_SIGMA}"

"""
Synthetic-data parameter recovery test for M2 (Student-T likelihood).

Generate observations from a Student-T with known nu, fit M2, assert that
the posterior puts substantial mass near the true nu and recovers taus.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.data_prep import Encoding
from src.models.bhm_heavytail import build_model


# True parameter values to recover.
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


def _generate_synthetic() -> tuple[pd.DataFrame, Encoding]:
    """Sample data with a Student-T noise distribution; return (df, encoding)."""
    rng = np.random.default_rng(RANDOM_SEED)
    alpha_v = TRUE_TAU_VESSEL * rng.standard_normal(N_VESSEL)
    alpha_b = TRUE_TAU_BERTH * rng.standard_normal(N_BERTH)
    alpha_s = TRUE_TAU_SERVICE * rng.standard_normal(N_SERVICE)

    v_idx = rng.integers(0, N_VESSEL, size=N_OBS)
    b_idx = rng.integers(0, N_BERTH, size=N_OBS)
    s_idx = rng.integers(0, N_SERVICE, size=N_OBS)
    mu = TRUE_ALPHA0 + alpha_v[v_idx] + alpha_b[b_idx] + alpha_s[s_idx]
    # Heavy-tailed noise (small nu => more outliers) is the signal the model
    # must detect to "win" over a plain Normal.
    log_y = mu + TRUE_SIGMA * rng.standard_t(df=TRUE_NU, size=N_OBS)

    df = pd.DataFrame({
        "vessel_idx": v_idx, "berth_idx": b_idx, "service_idx": s_idx,
        "log_service_time": log_y, "service_time_hours": np.exp(log_y),
    })
    enc = Encoding(
        vessel={f"V{i}": i for i in range(N_VESSEL)},
        berth={f"B{i}": i for i in range(N_BERTH)},
        service={f"S{i}": i for i in range(N_SERVICE)},
        n_vessel=N_VESSEL, n_berth=N_BERTH, n_service=N_SERVICE,
    )
    return df, enc


def test_m2_recovers_heavy_tails():
    """Posterior nu is plausibly heavy-tailed; taus and alpha0 recovered."""
    df, enc = _generate_synthetic()
    # Priors centered on the truth so the test isolates whether sampling can
    # recover the parameters rather than testing prior choices.
    model = build_model(
        df, enc, alpha0_mean=TRUE_ALPHA0, alpha0_sd=1.0,
        tau_halfnormal_sd=1.0, sigma_halfnormal_sd=1.0,
    )
    with model:
        idata = pm.sample(
            draws=1000, tune=1000, chains=2, target_accept=0.95,
            random_seed=RANDOM_SEED, progressbar=False,
        )

    # Any divergence means the posterior estimate may be biased; require zero.
    n_div = int(idata.sample_stats["diverging"].sum().item())
    assert n_div == 0, f"NUTS produced {n_div} divergences"

    for var, truth in [
        ("tau_vessel", TRUE_TAU_VESSEL),
        ("tau_berth", TRUE_TAU_BERTH),
        ("tau_service", TRUE_TAU_SERVICE),
        ("alpha0", TRUE_ALPHA0),
    ]:
        bounds = az.hdi(idata, var_names=[var], hdi_prob=HDI_PROB)[var].values
        assert bounds[0] <= truth <= bounds[1], f"{var} HDI {bounds} misses truth {truth}"

    # A small median nu confirms the model noticed the heavy tails we simulated.
    nu_med = float(np.median(idata.posterior["nu"].values))
    assert nu_med < 15, f"nu median {nu_med} suggests model failed to detect heavy tails"

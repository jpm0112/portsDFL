"""
Synthetic-data parameter recovery test for M3 (vessel-specific sigma).

Generate observations where each vessel type has its own sigma multiplier;
fit M3, assert that the posterior recovers the dispersion pattern.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.data_prep import Encoding
from src.models.bhm_heteroscedastic import build_model


# Each VESSEL TYPE gets its own noise spread, the heteroscedastic effect M3
# should recover.
N_VESSEL = 6
N_BERTH = 6
N_SERVICE = 10
N_OBS = 2000
RANDOM_SEED = 9090

TRUE_ALPHA0 = 3.5
TRUE_TAU_VESSEL = 0.4
TRUE_TAU_BERTH = 0.3
TRUE_TAU_SERVICE = 0.4
TRUE_SIGMA_GLOBAL = 0.4
# Vessel-specific sigma multipliers (multiplicative shocks).
TRUE_SIGMA_MULT = np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
HDI_PROB = 0.94


def _generate_synthetic() -> tuple[pd.DataFrame, Encoding]:
    rng = np.random.default_rng(RANDOM_SEED)
    alpha_v = TRUE_TAU_VESSEL * rng.standard_normal(N_VESSEL)
    alpha_b = TRUE_TAU_BERTH * rng.standard_normal(N_BERTH)
    alpha_s = TRUE_TAU_SERVICE * rng.standard_normal(N_SERVICE)
    sigma_v = TRUE_SIGMA_GLOBAL * TRUE_SIGMA_MULT  # per vessel sigma

    v_idx = rng.integers(0, N_VESSEL, size=N_OBS)
    b_idx = rng.integers(0, N_BERTH, size=N_OBS)
    s_idx = rng.integers(0, N_SERVICE, size=N_OBS)
    mu = TRUE_ALPHA0 + alpha_v[v_idx] + alpha_b[b_idx] + alpha_s[s_idx]
    # Noise spread depends on the row's vessel, so noisier vessels get wider scatter.
    log_y = mu + rng.normal(0.0, sigma_v[v_idx])

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


def test_m3_recovers_per_vessel_dispersion():
    """The posterior ranks vessel sigmas in the correct order with high probability."""
    df, enc = _generate_synthetic()
    # eta_sd=0.5 is a fairly loose prior on per-vessel log-sigma drift.
    model = build_model(
        df, enc,
        alpha0_mean=TRUE_ALPHA0, alpha0_sd=1.0,
        tau_halfnormal_sd=1.0, sigma_halfnormal_sd=1.0, eta_sd=0.5,
    )
    with model:
        idata = pm.sample(
            draws=1000, tune=1000, chains=2, target_accept=0.95,
            random_seed=RANDOM_SEED, progressbar=False,
        )

    # No divergences => the sampler explored the posterior reliably.
    n_div = int(idata.sample_stats["diverging"].sum().item())
    assert n_div == 0, f"NUTS produced {n_div} divergences"

    # sigma_vessel has N_VESSEL+1 entries (the last is the OOV/global fallback
    # slot), so [:N_VESSEL] drops it to align with `truth` below.
    sigma_v_post = idata.posterior["sigma_vessel"].mean(dim=("chain", "draw")).values[:N_VESSEL]
    # Allow small permutations: assert correlation with truth (~1.0 = correct ranking).
    truth = TRUE_SIGMA_GLOBAL * TRUE_SIGMA_MULT
    corr = float(np.corrcoef(sigma_v_post, truth)[0, 1])
    assert corr > 0.8, f"Recovered sigma_vessel correlation with truth = {corr:.3f}"

    # Loose band because the multiplicative per-vessel shocks make the global
    # level less identifiable.
    sg_mean = float(idata.posterior["sigma_global"].mean())
    assert 0.6 * TRUE_SIGMA_GLOBAL < sg_mean < 1.6 * TRUE_SIGMA_GLOBAL

"""
Synthetic-data parameter recovery test for M1 (M0 + linear covariates).

Generate data from the M1 generative process with known beta coefficients,
fit M1, and assert HDI containment for taus and beta means within tolerance.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.data_prep import CovariateScaler, Encoding
from src.models.bhm_covariates import build_model


N_VESSEL = 6
N_BERTH = 6
N_SERVICE = 10
N_OBS = 1500
RANDOM_SEED = 31415

TRUE_ALPHA0 = 3.5
TRUE_TAU_VESSEL = 0.4
TRUE_TAU_BERTH = 0.3
TRUE_TAU_SERVICE = 0.4
TRUE_SIGMA = 0.35
TRUE_BETA = np.array([0.30, 0.20, -0.05, 0.05, 0.0, 0.0, 0.10])  # 7 features
HDI_PROB = 0.94


def _generate_synthetic() -> tuple[pd.DataFrame, Encoding, CovariateScaler]:
    """Sample data with the M1 generative process; return (df, encoding, scaler)."""
    rng = np.random.default_rng(RANDOM_SEED)
    alpha_v = TRUE_TAU_VESSEL * rng.standard_normal(N_VESSEL)
    alpha_b = TRUE_TAU_BERTH * rng.standard_normal(N_BERTH)
    alpha_s = TRUE_TAU_SERVICE * rng.standard_normal(N_SERVICE)

    v_idx = rng.integers(0, N_VESSEL, size=N_OBS)
    b_idx = rng.integers(0, N_BERTH, size=N_OBS)
    s_idx = rng.integers(0, N_SERVICE, size=N_OBS)
    Z = rng.standard_normal(size=(N_OBS, len(TRUE_BETA)))

    mu = TRUE_ALPHA0 + alpha_v[v_idx] + alpha_b[b_idx] + alpha_s[s_idx] + Z @ TRUE_BETA
    log_y = mu + rng.normal(0.0, TRUE_SIGMA, size=N_OBS)

    feat = ["log_trg", "abs_calado_diff", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "year_trend"]
    cols = {f"z_{c}": Z[:, i] for i, c in enumerate(feat)}
    df = pd.DataFrame({
        "vessel_idx": v_idx, "berth_idx": b_idx, "service_idx": s_idx,
        "log_service_time": log_y, "service_time_hours": np.exp(log_y),
        **cols,
    })
    enc = Encoding(
        vessel={f"V{i}": i for i in range(N_VESSEL)},
        berth={f"B{i}": i for i in range(N_BERTH)},
        service={f"S{i}": i for i in range(N_SERVICE)},
        n_vessel=N_VESSEL, n_berth=N_BERTH, n_service=N_SERVICE,
    )
    scaler = CovariateScaler(
        means={c: 0.0 for c in feat},
        stds={c: 1.0 for c in feat},
        feature_cols=feat,
        beta_names=[f"beta_{c}" for c in feat],
    )
    return df, enc, scaler


def test_m1_recovers_beta_and_taus():
    """Posterior HDIs cover true taus and beta means; sigma is tight."""
    df, enc, scaler = _generate_synthetic()
    model = build_model(
        df, enc, scaler=scaler,
        alpha0_mean=TRUE_ALPHA0, alpha0_sd=1.0,
        tau_halfnormal_sd=1.0, sigma_halfnormal_sd=1.0, beta_sd=1.0,
    )
    with model:
        idata = pm.sample(
            draws=1000, tune=1000, chains=2, target_accept=0.95,
            random_seed=RANDOM_SEED, progressbar=False,
        )

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

    # Beta posterior means should be close to truth.
    beta_post = idata.posterior["beta"].mean(dim=("chain", "draw")).values
    np.testing.assert_allclose(beta_post, TRUE_BETA, atol=0.07)

    sigma_mean = float(idata.posterior["sigma"].mean())
    assert abs(sigma_mean - TRUE_SIGMA) < 0.04

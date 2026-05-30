"""
Synthetic-data parameter recovery test for M1 (M0 + linear covariates).

Generate data from the M1 generative process with known beta coefficients,
fit M1, and assert HDI containment for taus and beta means within tolerance.
"""

# See test_bhm_baseline.py for first-time explanations of __future__ annotations,
# the arviz/numpy/pandas/pymc imports, type hints, f-strings, comprehensions,
# `with pm.Model()`, pm.sample, HDI, and the parameter-recovery idea. This file
# extends that test to the M1 model (M0 baseline plus linear covariates).
from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

# CovariateScaler stores the mean/std used to standardize each covariate, plus
# the ordered list of feature columns. The model reads betas in that same order.
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
# True coefficient for each of the 7 covariates. Two are exactly 0.0, which
# checks the model can correctly conclude "no effect" for some features too.
# The ORDER here must line up with the feature order built below.
TRUE_BETA = np.array([0.30, 0.20, -0.05, 0.05, 0.0, 0.0, 0.10])  # 7 features
HDI_PROB = 0.94


def _generate_synthetic() -> tuple[pd.DataFrame, Encoding, CovariateScaler]:
    """Sample data with the M1 generative process; return (df, encoding, scaler)."""
    rng = np.random.default_rng(RANDOM_SEED)
    # Group offsets, same non-centered (tau * standard-normal) construction as M0.
    alpha_v = TRUE_TAU_VESSEL * rng.standard_normal(N_VESSEL)
    alpha_b = TRUE_TAU_BERTH * rng.standard_normal(N_BERTH)
    alpha_s = TRUE_TAU_SERVICE * rng.standard_normal(N_SERVICE)

    v_idx = rng.integers(0, N_VESSEL, size=N_OBS)
    b_idx = rng.integers(0, N_BERTH, size=N_OBS)
    s_idx = rng.integers(0, N_SERVICE, size=N_OBS)
    # Z is the covariate matrix: one row per observation, one column per feature.
    # Drawn standard-normal because the model expects already-standardized inputs.
    Z = rng.standard_normal(size=(N_OBS, len(TRUE_BETA)))

    # `Z @ TRUE_BETA` is matrix multiplication (the @ operator): for each row it
    # computes the dot product sum(feature_value * coefficient), i.e. the total
    # linear-covariate contribution to that observation's expected log target.
    mu = TRUE_ALPHA0 + alpha_v[v_idx] + alpha_b[b_idx] + alpha_s[s_idx] + Z @ TRUE_BETA
    log_y = mu + rng.normal(0.0, TRUE_SIGMA, size=N_OBS)

    feat = ["log_trg", "abs_calado_diff", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "year_trend"]
    # Build the standardized-covariate columns the model reads (named "z_<feature>").
    # enumerate(feat) yields (i, c) pairs: i is the column position, c the name,
    # so Z[:, i] (column i of Z) becomes column "z_c". This keeps Z's column
    # order aligned with `feat`, which must match TRUE_BETA's order.
    cols = {f"z_{c}": Z[:, i] for i, c in enumerate(feat)}
    # `**cols` unpacks the z_* covariate dict into this DataFrame literal, so all
    # 7 standardized covariate columns are added alongside the index/target columns.
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
    # Identity scaler: means 0 and stds 1 for every feature, so standardization
    # is a no-op. We already drew Z standard-normal, so the z_* columns ARE the
    # raw covariates and beta_sd stays directly interpretable.
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
    # build_model for M1 adds a `beta` vector (one coefficient per covariate)
    # with a Normal(0, beta_sd) prior on top of the M0 hierarchy.
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

    # Require a clean sample: no divergences (see baseline test for why).
    n_div = int(idata.sample_stats["diverging"].sum().item())
    assert n_div == 0, f"NUTS produced {n_div} divergences"

    # Loop over (parameter name, true value) pairs and check each 94% HDI
    # brackets its truth. Looping keeps the four near-identical checks compact.
    for var, truth in [
        ("tau_vessel", TRUE_TAU_VESSEL),
        ("tau_berth", TRUE_TAU_BERTH),
        ("tau_service", TRUE_TAU_SERVICE),
        ("alpha0", TRUE_ALPHA0),
    ]:
        bounds = az.hdi(idata, var_names=[var], hdi_prob=HDI_PROB)[var].values
        assert bounds[0] <= truth <= bounds[1], f"{var} HDI {bounds} misses truth {truth}"

    # Beta posterior means should be close to truth.
    # `beta` is a vector parameter; averaging over the chain and draw dimensions
    # collapses every posterior sample into one mean coefficient per feature.
    beta_post = idata.posterior["beta"].mean(dim=("chain", "draw")).values
    # assert_allclose checks each recovered coefficient is within atol of truth;
    # atol=0.07 is the absolute tolerance (also passes for the two true-zero betas).
    np.testing.assert_allclose(beta_post, TRUE_BETA, atol=0.07)

    # Residual sd should likewise be recovered tightly given 1500 observations.
    sigma_mean = float(idata.posterior["sigma"].mean())
    assert abs(sigma_mean - TRUE_SIGMA) < 0.04

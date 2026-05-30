"""
Synthetic-data parameter recovery test for M4 (selective vessel * berth interactions).

Generate observations with planted interaction effects in well-populated
(vessel, berth) cells; fit M4 with min_interaction_n appropriate to the
synthetic data; assert that tau_vb is meaningfully > 0 and main-effect
taus are still recovered.
"""

# See test_bhm_baseline.py for first-time explanations of the imports, type hints,
# f-strings, comprehensions, `with pm.Model()`, pm.sample, HDI, and the
# parameter-recovery approach. This file tests M4: M0 plus a vessel*berth
# interaction term (an extra offset for each specific vessel/berth combination).
from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from src.data_prep import Encoding
from src.models.bhm_interactions import build_model


N_VESSEL = 4
N_BERTH = 4
N_SERVICE = 8
N_OBS = 1500
RANDOM_SEED = 7777

TRUE_ALPHA0 = 3.5
TRUE_TAU_VESSEL = 0.4
TRUE_TAU_BERTH = 0.3
TRUE_TAU_SERVICE = 0.4
# TRUE_TAU_VB is the spread of the vessel*berth interaction effects. We plant a
# clearly non-zero value so a correct model should detect that interactions exist.
TRUE_TAU_VB = 0.30
TRUE_SIGMA = 0.35
HDI_PROB = 0.94


def _generate_synthetic() -> tuple[pd.DataFrame, Encoding]:
    """Sample data with non-zero interaction terms in every (vessel, berth) cell."""
    rng = np.random.default_rng(RANDOM_SEED)
    # Main-effect group offsets, same non-centered construction as M0.
    alpha_v = TRUE_TAU_VESSEL * rng.standard_normal(N_VESSEL)
    alpha_b = TRUE_TAU_BERTH * rng.standard_normal(N_BERTH)
    alpha_s = TRUE_TAU_SERVICE * rng.standard_normal(N_SERVICE)
    # Interaction offsets: a full N_VESSEL x N_BERTH grid, one extra effect for
    # every (vessel, berth) pair. This is the structure M4 is meant to recover.
    gamma_vb = TRUE_TAU_VB * rng.standard_normal((N_VESSEL, N_BERTH))

    v_idx = rng.integers(0, N_VESSEL, size=N_OBS)
    b_idx = rng.integers(0, N_BERTH, size=N_OBS)
    s_idx = rng.integers(0, N_SERVICE, size=N_OBS)
    # `gamma_vb[v_idx, b_idx]` is paired 2-D fancy indexing: for each observation
    # it looks up the interaction offset for that observation's (vessel, berth)
    # cell, giving a 1-D array of length N_OBS.
    mu = (
        TRUE_ALPHA0 + alpha_v[v_idx] + alpha_b[b_idx] + alpha_s[s_idx]
        + gamma_vb[v_idx, b_idx]
    )
    log_y = mu + rng.normal(0.0, TRUE_SIGMA, size=N_OBS)

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


def test_m4_recovers_interactions():
    """tau_vb posterior is meaningfully above zero; main-effect taus recovered."""
    df, enc = _generate_synthetic()
    # Use a low threshold so most cells in the synthetic 4x4 grid get an interaction term.
    # min_interaction_n=20: only (vessel, berth) cells with at least 20 training
    # rows get their own interaction parameter; sparser cells fall back to the
    # additive M0 structure. With 1500 rows spread over 16 cells (~94 each),
    # essentially every cell clears the bar, so all planted effects are estimable.
    model = build_model(
        df, enc, alpha0_mean=TRUE_ALPHA0, alpha0_sd=1.0,
        tau_halfnormal_sd=1.0, sigma_halfnormal_sd=1.0,
        min_interaction_n=20,
    )
    with model:
        idata = pm.sample(
            draws=1000, tune=1000, chains=2, target_accept=0.95,
            random_seed=RANDOM_SEED, progressbar=False,
        )

    # Require a clean sample: no divergences (see baseline test for why).
    n_div = int(idata.sample_stats["diverging"].sum().item())
    assert n_div == 0, f"NUTS produced {n_div} divergences"

    # tau_vb posterior 5th percentile should be above 0.05 (i.e., the model is using interactions).
    # np.quantile(values, 0.05) is the value below which 5% of the posterior
    # samples fall. If even that low end sits above 0.05, the posterior is
    # confidently away from zero, meaning the model detected real interactions.
    tau_vb_q05 = float(np.quantile(idata.posterior["tau_vb"].values, 0.05))
    assert tau_vb_q05 > 0.05, f"tau_vb 5th percentile {tau_vb_q05} suggests model didn't pick up interactions"

    # Adding interactions must not break recovery of the main-effect parameters:
    # each 94% HDI should still bracket its true value.
    for var, truth in [
        ("tau_vessel", TRUE_TAU_VESSEL),
        ("tau_berth", TRUE_TAU_BERTH),
        ("tau_service", TRUE_TAU_SERVICE),
        ("alpha0", TRUE_ALPHA0),
    ]:
        bounds = az.hdi(idata, var_names=[var], hdi_prob=HDI_PROB)[var].values
        assert bounds[0] <= truth <= bounds[1], f"{var} HDI {bounds} misses truth {truth}"

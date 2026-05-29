"""
M3: M0 with vessel-type-specific residual sigma (heteroscedastic).

Different vessel types have visibly different log(svc) spreads in the M0
PPC. A single global sigma over-shrinks Container variance and over-
inflates Passenger variance. Letting sigma vary by vessel type fixes
this without adding much complexity.

Likelihood:
    log(svc_i)       ~ Normal(mu_i, sigma_vessel[v(i)])
    sigma_vessel[k]  = sigma_global * exp(eta_v[k])    (non-centered)
    eta_v[k]         ~ Normal(0, eta_sd)
    sigma_global     ~ HalfNormal(sigma_halfnormal_sd)

This is the canonical "varying-scale" hierarchical Lognormal. The eta
parameterization makes per-vessel sigmas multiplicative shocks around the
global sigma, with eta_sd controlling how much they can differ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm

from ..data_prep import OOV_INDEX, Encoding
from .bhm_baseline import _remap_with_oov


def build_model(
    train_df: pd.DataFrame,
    encoding: Encoding,
    scaler=None,  # unused
    alpha0_mean: float = 3.47,
    alpha0_sd: float = 1.0,
    tau_halfnormal_sd: float = 0.5,
    sigma_halfnormal_sd: float = 0.7,
    eta_sd: float = 0.4,
    **_unused,
) -> pm.Model:
    """
    Build M3: hierarchical Lognormal with vessel-specific residual sigma.

    Input:
        Same as M0, plus eta_sd controlling the prior spread of vessel-specific
        log-sigma shocks. eta_sd=0.4 allows per-vessel sigma to vary by ~+/-50%.

    Output: pm.Model.
    """
    coords = {
        "vessel": list(encoding.vessel.keys()),
        "berth": list(encoding.berth.keys()),
        "service": list(encoding.service.keys()),
    }

    v_idx = _remap_with_oov(train_df["vessel_idx"].to_numpy(), encoding.n_vessel)
    b_idx = _remap_with_oov(train_df["berth_idx"].to_numpy(), encoding.n_berth)
    s_idx = _remap_with_oov(train_df["service_idx"].to_numpy(), encoding.n_service)
    log_y = train_df["log_service_time"].to_numpy()

    with pm.Model(coords=coords) as model:
        vessel_idx_data = pm.MutableData("vessel_idx", v_idx)
        berth_idx_data = pm.MutableData("berth_idx", b_idx)
        service_idx_data = pm.MutableData("service_idx", s_idx)
        log_y_data = pm.MutableData("log_y", log_y)

        alpha0 = pm.Normal("alpha0", mu=alpha0_mean, sigma=alpha0_sd)
        tau_vessel = pm.HalfNormal("tau_vessel", sigma=tau_halfnormal_sd)
        tau_berth = pm.HalfNormal("tau_berth", sigma=tau_halfnormal_sd)
        tau_service = pm.HalfNormal("tau_service", sigma=tau_halfnormal_sd)

        z_vessel = pm.Normal("z_vessel", mu=0.0, sigma=1.0, dims="vessel")
        z_berth = pm.Normal("z_berth", mu=0.0, sigma=1.0, dims="berth")
        z_service = pm.Normal("z_service", mu=0.0, sigma=1.0, dims="service")

        alpha_vessel = pm.Deterministic("alpha_vessel", pm.math.concatenate([tau_vessel * z_vessel, [0.0]]))
        alpha_berth = pm.Deterministic("alpha_berth", pm.math.concatenate([tau_berth * z_berth, [0.0]]))
        alpha_service = pm.Deterministic("alpha_service", pm.math.concatenate([tau_service * z_service, [0.0]]))

        # Vessel-specific log-sigma shocks. The OOV slot uses zero shock
        # (i.e., the global sigma) since we have no information.
        sigma_global = pm.HalfNormal("sigma_global", sigma=sigma_halfnormal_sd)
        eta_v = pm.Normal("eta_v", mu=0.0, sigma=eta_sd, dims="vessel")
        sigma_vessel_core = sigma_global * pm.math.exp(eta_v)
        sigma_vessel = pm.Deterministic(
            "sigma_vessel", pm.math.concatenate([sigma_vessel_core, [sigma_global]])
        )

        mu = (
            alpha0
            + alpha_vessel[vessel_idx_data]
            + alpha_berth[berth_idx_data]
            + alpha_service[service_idx_data]
        )
        pm.Normal(
            "log_y_obs", mu=mu, sigma=sigma_vessel[vessel_idx_data], observed=log_y_data
        )

    return model

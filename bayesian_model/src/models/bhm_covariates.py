"""
M1: M0 + linear covariates.

Adds standardized log(TRG), |Calado diff|, hour and day-of-week cyclic
encodings, and a year trend. Each beta is given a Normal(0, beta_sd)
prior on the standardized scale, so beta_sd is directly interpretable as
"plausible effect of one SD of the covariate on log(svc)".

Likelihood and hierarchy are unchanged from M0:
    log(svc_i) ~ Normal(mu_i, sigma)
    mu_i = alpha0 + alpha_vessel + alpha_berth + alpha_service + Z_i @ beta
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm

from ..data_prep import OOV_INDEX, CovariateScaler, Encoding
from .bhm_baseline import _remap_with_oov


def build_model(
    train_df: pd.DataFrame,
    encoding: Encoding,
    scaler: CovariateScaler,
    alpha0_mean: float = 3.47,
    alpha0_sd: float = 1.0,
    tau_halfnormal_sd: float = 0.5,
    sigma_halfnormal_sd: float = 0.7,
    beta_sd: float = 0.5,
    **_unused,
) -> pm.Model:
    """
    Build M1: hierarchical Lognormal with linear covariates.

    Input:
        train_df, encoding, scaler: produced by data_prep.prepare(with_covariates=True).
        alpha0_*, tau_halfnormal_sd, sigma_halfnormal_sd: same as M0.
        beta_sd: prior standard deviation on each beta coefficient (Normal(0, beta_sd)).
                 Default 0.5 implies a plausible effect of ~+/-1 log-unit (factor of e)
                 from a +/-2 SD swing in the covariate, which is generous.

    Output:
        pm.Model with the same hierarchical structure as M0 plus a vector
        of betas indexed by the standardized feature columns from scaler.

    Description:
        Covariates are taken from train_df columns z_<feature> for
        `feature` in scaler.feature_cols. They are wrapped in pm.MutableData
        so the model can be reused at predict time with new rows.
    """
    if scaler is None:
        raise ValueError("M1 requires a CovariateScaler. Call data_prep.prepare(with_covariates=True).")

    coords = {
        "vessel": list(encoding.vessel.keys()),
        "berth": list(encoding.berth.keys()),
        "service": list(encoding.service.keys()),
        "feature": scaler.feature_cols,  # one beta coefficient per standardized covariate
    }

    v_idx = _remap_with_oov(train_df["vessel_idx"].to_numpy(), encoding.n_vessel)
    b_idx = _remap_with_oov(train_df["berth_idx"].to_numpy(), encoding.n_berth)
    s_idx = _remap_with_oov(train_df["service_idx"].to_numpy(), encoding.n_service)
    log_y = train_df["log_service_time"].to_numpy()
    Z = train_df[[f"z_{c}" for c in scaler.feature_cols]].to_numpy()

    with pm.Model(coords=coords) as model:
        # Mutable inputs so the same compiled model can be reused at predict time.
        vessel_idx_data = pm.Data("vessel_idx", v_idx)
        berth_idx_data = pm.Data("berth_idx", b_idx)
        service_idx_data = pm.Data("service_idx", s_idx)
        Z_data = pm.Data("Z", Z, dims=("obs", "feature"))
        log_y_data = pm.Data("log_y", log_y)

        # Global intercept on the log scale.
        alpha0 = pm.Normal("alpha0", mu=alpha0_mean, sigma=alpha0_sd)
        tau_vessel = pm.HalfNormal("tau_vessel", sigma=tau_halfnormal_sd)
        tau_berth = pm.HalfNormal("tau_berth", sigma=tau_halfnormal_sd)
        tau_service = pm.HalfNormal("tau_service", sigma=tau_halfnormal_sd)

        # Non-centered parameterization avoids the funnel geometry that makes
        # NUTS sampling unreliable.
        z_vessel = pm.Normal("z_vessel", mu=0.0, sigma=1.0, dims="vessel")
        z_berth = pm.Normal("z_berth", mu=0.0, sigma=1.0, dims="berth")
        z_service = pm.Normal("z_service", mu=0.0, sigma=1.0, dims="service")

        # Trailing 0.0 is the zero-offset slot OOV rows were remapped to.
        alpha_vessel = pm.Deterministic("alpha_vessel", pm.math.concatenate([tau_vessel * z_vessel, [0.0]]))
        alpha_berth = pm.Deterministic("alpha_berth", pm.math.concatenate([tau_berth * z_berth, [0.0]]))
        alpha_service = pm.Deterministic("alpha_service", pm.math.concatenate([tau_service * z_service, [0.0]]))

        # Covariates are on the SD scale, so beta_sd is the prior "effect of a
        # 1-SD change" on log(svc).
        beta = pm.Normal("beta", mu=0.0, sigma=beta_sd, dims="feature")

        mu = (
            alpha0
            + alpha_vessel[vessel_idx_data]
            + alpha_berth[berth_idx_data]
            + alpha_service[service_idx_data]
            + pm.math.dot(Z_data, beta)
        )
        # Residual log-scale standard deviation (Lognormal -> Normal on log y).
        sigma = pm.HalfNormal("sigma", sigma=sigma_halfnormal_sd)
        pm.Normal("log_y_obs", mu=mu, sigma=sigma, observed=log_y_data)

    return model

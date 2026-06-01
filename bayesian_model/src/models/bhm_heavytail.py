"""
M2: M0 with a Student-T (heavy-tail) likelihood on log(svc).

The PIT histogram for M0 showed a spike at 0, indicating that the upper
tail of service time is underpredicted by a Lognormal. Replacing the
Normal observation with a Student-T allows occasional large residuals
without inflating the global sigma. The degrees-of-freedom parameter
controls how heavy the tails are (df -> infinity recovers Normal).

Likelihood:
    log(svc_i) ~ StudentT(nu, mu_i, sigma)
    nu - 1     ~ Gamma(2, 0.1)        # nu in [1, ~70] with prior mean ~21

Hierarchy: identical to M0.
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
    nu_alpha: float = 2.0,
    nu_beta: float = 0.1,
    **_unused,
) -> pm.Model:
    """
    Build M2: hierarchical Student-T regression on log(svc).

    Input:
        train_df, encoding: as in M0.
        alpha0_*, tau_halfnormal_sd, sigma_halfnormal_sd: as in M0.
        nu_alpha, nu_beta: shape and rate of the Gamma prior on (nu - 1).
                           Defaults give E[nu] ~21, plenty of mass on heavy
                           tails (nu < 5) while still allowing near-Normal.

    Output: pm.Model.

    Description:
        Same partial-pooling structure on the location parameter mu_i;
        only the noise distribution changes. The Gamma(2, 0.1) prior on
        (nu - 1) is the standard weakly-informative choice from Juarez &
        Steel 2010 (BAS).
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
        # Mutable inputs so the same compiled model can be reused at predict time.
        vessel_idx_data = pm.Data("vessel_idx", v_idx)
        berth_idx_data = pm.Data("berth_idx", b_idx)
        service_idx_data = pm.Data("service_idx", s_idx)
        log_y_data = pm.Data("log_y", log_y)

        # Global intercept on the log scale.
        alpha0 = pm.Normal("alpha0", mu=alpha0_mean, sigma=alpha0_sd)
        tau_vessel = pm.HalfNormal("tau_vessel", sigma=tau_halfnormal_sd)
        tau_berth = pm.HalfNormal("tau_berth", sigma=tau_halfnormal_sd)
        tau_service = pm.HalfNormal("tau_service", sigma=tau_halfnormal_sd)

        # Non-centered parameterization (see bhm_baseline for the funnel rationale).
        z_vessel = pm.Normal("z_vessel", mu=0.0, sigma=1.0, dims="vessel")
        z_berth = pm.Normal("z_berth", mu=0.0, sigma=1.0, dims="berth")
        z_service = pm.Normal("z_service", mu=0.0, sigma=1.0, dims="service")

        # Trailing 0.0 is the OOV slot (zero offset).
        alpha_vessel = pm.Deterministic("alpha_vessel", pm.math.concatenate([tau_vessel * z_vessel, [0.0]]))
        alpha_berth = pm.Deterministic("alpha_berth", pm.math.concatenate([tau_berth * z_berth, [0.0]]))
        alpha_service = pm.Deterministic("alpha_service", pm.math.concatenate([tau_service * z_service, [0.0]]))

        mu = (
            alpha0
            + alpha_vessel[vessel_idx_data]
            + alpha_berth[berth_idx_data]
            + alpha_service[service_idx_data]
        )
        sigma = pm.HalfNormal("sigma", sigma=sigma_halfnormal_sd)

        # Reparameterize nu so it stays > 1. Gamma(2, 0.1) gives nu a prior mean
        # ~21 (near-Normal but with room for heavy tails); small nu -> heavier tails.
        nu_minus_one = pm.Gamma("nu_minus_one", alpha=nu_alpha, beta=nu_beta)
        nu = pm.Deterministic("nu", nu_minus_one + 1.0)

        # StudentT (vs Normal in M0) lets occasional large residuals occur
        # without inflating sigma -- the heavy-tail fix this model is about.
        pm.StudentT("log_y_obs", nu=nu, mu=mu, sigma=sigma, observed=log_y_data)

    return model

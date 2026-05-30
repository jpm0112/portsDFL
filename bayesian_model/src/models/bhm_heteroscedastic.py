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

# `from __future__ import annotations` makes type hints (e.g. `X | None`) be
# treated as text so they work on older Python. Must be the first import.
from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm  # PyMC: the Bayesian modelling / MCMC library

# `..data_prep` = up one package level. OOV_INDEX (-1) marks unseen categories;
# Encoding holds the string->int lookup tables.
from ..data_prep import OOV_INDEX, Encoding
# Reuse the baseline's helper that routes OOV (-1) codes to the zero slot.
from .bhm_baseline import _remap_with_oov


# `def f(...) -> pm.Model:` defines a function returning a built PyMC model.
# Each `name: type = default` argument is optional (callers may omit it).
def build_model(
    train_df: pd.DataFrame,
    encoding: Encoding,
    scaler=None,  # unused
    alpha0_mean: float = 3.47,
    alpha0_sd: float = 1.0,
    tau_halfnormal_sd: float = 0.5,
    sigma_halfnormal_sd: float = 0.7,
    eta_sd: float = 0.4,
    # `**_unused` swallows any extra keyword args so all models share one
    # uniform call signature.
    **_unused,
) -> pm.Model:
    """
    Build M3: hierarchical Lognormal with vessel-specific residual sigma.

    Input:
        Same as M0, plus eta_sd controlling the prior spread of vessel-specific
        log-sigma shocks. eta_sd=0.4 allows per-vessel sigma to vary by ~+/-50%.

    Output: pm.Model.
    """
    # `coords` names each model dimension and lists its category labels so
    # PyMC/ArviZ can tag outputs by name. `.keys()` -> the strings; `list(...)`
    # makes a plain list.
    coords = {
        "vessel": list(encoding.vessel.keys()),
        "berth": list(encoding.berth.keys()),
        "service": list(encoding.service.keys()),
    }

    # Integer codes per factor as NumPy arrays, with OOV (-1) routed to the
    # appended zero slot at position n_levels.
    v_idx = _remap_with_oov(train_df["vessel_idx"].to_numpy(), encoding.n_vessel)
    b_idx = _remap_with_oov(train_df["berth_idx"].to_numpy(), encoding.n_berth)
    s_idx = _remap_with_oov(train_df["service_idx"].to_numpy(), encoding.n_service)
    log_y = train_df["log_service_time"].to_numpy()  # observed targets, log(svc)

    # Context manager: random variables defined in this block are registered on
    # `model`. pm.MutableData wraps inputs so they can be swapped at predict time.
    with pm.Model(coords=coords) as model:
        vessel_idx_data = pm.MutableData("vessel_idx", v_idx)
        berth_idx_data = pm.MutableData("berth_idx", b_idx)
        service_idx_data = pm.MutableData("service_idx", s_idx)
        log_y_data = pm.MutableData("log_y", log_y)

        # Priors: global intercept (Normal) and group scales (HalfNormal >= 0).
        alpha0 = pm.Normal("alpha0", mu=alpha0_mean, sigma=alpha0_sd)
        tau_vessel = pm.HalfNormal("tau_vessel", sigma=tau_halfnormal_sd)
        tau_berth = pm.HalfNormal("tau_berth", sigma=tau_halfnormal_sd)
        tau_service = pm.HalfNormal("tau_service", sigma=tau_halfnormal_sd)

        # Non-centered offsets: standard normals (one per category) scaled by tau.
        z_vessel = pm.Normal("z_vessel", mu=0.0, sigma=1.0, dims="vessel")
        z_berth = pm.Normal("z_berth", mu=0.0, sigma=1.0, dims="berth")
        z_service = pm.Normal("z_service", mu=0.0, sigma=1.0, dims="service")

        # pm.Deterministic saves a derived quantity in the trace. Group offsets
        # = tau * z, with a trailing 0.0 as the OOV slot (zero offset).
        alpha_vessel = pm.Deterministic("alpha_vessel", pm.math.concatenate([tau_vessel * z_vessel, [0.0]]))
        alpha_berth = pm.Deterministic("alpha_berth", pm.math.concatenate([tau_berth * z_berth, [0.0]]))
        alpha_service = pm.Deterministic("alpha_service", pm.math.concatenate([tau_service * z_service, [0.0]]))

        # Vessel-specific log-sigma shocks. The OOV slot uses zero shock
        # (i.e., the global sigma) since we have no information.
        sigma_global = pm.HalfNormal("sigma_global", sigma=sigma_halfnormal_sd)  # baseline noise scale
        # One log-shock per vessel type; eta_sd controls how far per-vessel
        # sigmas may drift from sigma_global.
        eta_v = pm.Normal("eta_v", mu=0.0, sigma=eta_sd, dims="vessel")
        # Multiplicative shock: exp(eta) > 0, so sigma_global * exp(eta) keeps
        # each per-vessel sigma positive (this is the heteroscedastic part).
        sigma_vessel_core = sigma_global * pm.math.exp(eta_v)
        # Append sigma_global at the OOV slot (index n_vessel) so unseen vessels
        # fall back to the global noise scale.
        sigma_vessel = pm.Deterministic(
            "sigma_vessel", pm.math.concatenate([sigma_vessel_core, [sigma_global]])
        )

        # Linear predictor (location mu): indexing each offset array by the
        # row's integer code selects that row's group effect.
        mu = (
            alpha0
            + alpha_vessel[vessel_idx_data]
            + alpha_berth[berth_idx_data]
            + alpha_service[service_idx_data]
        )
        # Likelihood: `observed=` binds this Normal to the data. Unlike M0, the
        # noise sigma varies per row via sigma_vessel indexed by vessel code --
        # the non-constant variance (heteroscedasticity) this model adds.
        pm.Normal(
            "log_y_obs", mu=mu, sigma=sigma_vessel[vessel_idx_data], observed=log_y_data
        )

    # Return the built model; MCMC sampling (pm.sample) happens in fit.py.
    return model

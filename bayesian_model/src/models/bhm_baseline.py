"""
M0 baseline Bayesian Hierarchical Model for vessel service time.

Likelihood:
    log(service_time_i) ~ Normal(mu_i, sigma)
    mu_i = alpha0 + alpha_vessel[v(i)] + alpha_berth[b(i)] + alpha_service[s(i)]

Hierarchy (non-centered to avoid the funnel pathology in NUTS):
    alpha_g[k] = tau_g * z_g[k]   with z_g[k] ~ Normal(0, 1)
    tau_g      ~ HalfNormal(tau_sd)        for g in {vessel, berth, service}
    alpha0     ~ Normal(alpha0_mean, alpha0_sd)
    sigma      ~ HalfNormal(sigma_sd)

Each tau_g controls how strongly the corresponding factor's group-level
effects shrink toward the global intercept. Sparse categories borrow
strength from well-populated ones because all share the same tau_g.

Out-of-vocabulary handling: indices equal to OOV_INDEX (-1) get a zero
group offset (i.e., they predict at alpha0 plus the offsets of any other
known factors). This is implemented by appending a constant-zero row to
each alpha_g vector and rewriting OOV indices to point at it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm

# OOV_INDEX is the sentinel (-1) for out-of-vocabulary categories.
from ..data_prep import OOV_INDEX, Encoding


def _remap_with_oov(idx: np.ndarray, n_levels: int) -> np.ndarray:
    """
    Remap OOV_INDEX (-1) to the appended zero-offset slot at position n_levels.

    Input:
        idx: integer array of factor indices, possibly containing -1.
        n_levels: number of trained levels for this factor.

    Output:
        Integer array with -1 replaced by n_levels.

    Description:
        The model defines n_levels learnable group offsets and then appends
        a constant zero. Any OOV row is routed to that zero slot so the
        prediction collapses to alpha0 + offsets from the known factors.
    """
    # OOV rows -> the appended zero-offset slot at position n_levels.
    out = np.where(idx == OOV_INDEX, n_levels, idx).astype("int64")
    return out


# **_unused lets every model in the registry accept the same call shape even
# if some priors don't apply to M0.
def build_model(
    train_df: pd.DataFrame,
    encoding: Encoding,
    scaler=None,  # accepted for uniform registry signature; unused by M0
    alpha0_mean: float = 3.47,
    alpha0_sd: float = 1.0,
    tau_halfnormal_sd: float = 0.5,
    sigma_halfnormal_sd: float = 0.7,
    **_unused,
) -> pm.Model:
    """
    Construct the M0 PyMC model bound to the training rows.

    Input:
        train_df: training rows with vessel_idx, berth_idx, service_idx,
                  log_service_time columns (produced by data_prep.prepare).
        encoding: Encoding dataclass holding n_vessel, n_berth, n_service.
        alpha0_mean / alpha0_sd: prior on the global intercept of log(svc).
        tau_halfnormal_sd: scale of HalfNormal prior on each group tau.
        sigma_halfnormal_sd: scale of HalfNormal prior on residual sigma.

    Output:
        pymc.Model ready for sampling. Group offsets are stored as
        Deterministics named alpha_vessel/alpha_berth/alpha_service, each
        with an extra zero row appended at index n_levels for OOV routing.

    Description:
        Uses pm.MutableData so that posterior_predictive.py can swap in
        new (vessel, berth, service) index arrays at predict time without
        rebuilding the model. log(svc) observations live in mu_obs and
        log_y_obs, both mutable.
    """
    # Human-readable labels per dimension so the posterior is tagged by
    # category name rather than integer position.
    coords = {
        "vessel": list(encoding.vessel.keys()),
        "berth": list(encoding.berth.keys()),
        "service": list(encoding.service.keys()),
    }

    v_idx = _remap_with_oov(train_df["vessel_idx"].to_numpy(), encoding.n_vessel)
    b_idx = _remap_with_oov(train_df["berth_idx"].to_numpy(), encoding.n_berth)
    s_idx = _remap_with_oov(train_df["service_idx"].to_numpy(), encoding.n_service)
    # Modelling on the log scale turns multiplicative spread into additive
    # Normal noise (a Lognormal y).
    log_y = train_df["log_service_time"].to_numpy()

    with pm.Model(coords=coords) as model:
        # Mutable inputs so the same compiled model can be reused for
        # posterior predictive draws on arbitrary cells via pm.set_data.
        vessel_idx_data = pm.Data("vessel_idx", v_idx)
        berth_idx_data = pm.Data("berth_idx", b_idx)
        service_idx_data = pm.Data("service_idx", s_idx)
        log_y_data = pm.Data("log_y", log_y)

        # Global intercept on the log scale.
        alpha0 = pm.Normal("alpha0", mu=alpha0_mean, sigma=alpha0_sd)

        # Group-level standard deviations: small tau => strong shrinkage/pooling
        # toward the global intercept.
        tau_vessel = pm.HalfNormal("tau_vessel", sigma=tau_halfnormal_sd)
        tau_berth = pm.HalfNormal("tau_berth", sigma=tau_halfnormal_sd)
        tau_service = pm.HalfNormal("tau_service", sigma=tau_halfnormal_sd)

        # Non-centered parameterization: sample standard normals and scale by
        # tau. Reshapes the posterior geometry so NUTS does not get stuck in
        # the "funnel" near tau ~ 0.
        z_vessel = pm.Normal("z_vessel", mu=0.0, sigma=1.0, dims="vessel")
        z_berth = pm.Normal("z_berth", mu=0.0, sigma=1.0, dims="berth")
        z_service = pm.Normal("z_service", mu=0.0, sigma=1.0, dims="service")

        alpha_vessel_core = tau_vessel * z_vessel
        alpha_berth_core = tau_berth * z_berth
        alpha_service_core = tau_service * z_service

        # Append a constant 0.0 as the slot OOV rows (remapped to n_levels)
        # point at, so an unknown category contributes a zero offset.
        alpha_vessel = pm.Deterministic(
            "alpha_vessel", pm.math.concatenate([alpha_vessel_core, [0.0]])
        )
        alpha_berth = pm.Deterministic(
            "alpha_berth", pm.math.concatenate([alpha_berth_core, [0.0]])
        )
        alpha_service = pm.Deterministic(
            "alpha_service", pm.math.concatenate([alpha_service_core, [0.0]])
        )

        # Expected log(service_time) per row.
        mu = (
            alpha0
            + alpha_vessel[vessel_idx_data]
            + alpha_berth[berth_idx_data]
            + alpha_service[service_idx_data]
        )

        # Residual log-scale standard deviation (Lognormal -> Normal on log y).
        sigma = pm.HalfNormal("sigma", sigma=sigma_halfnormal_sd)

        pm.Normal("log_y_obs", mu=mu, sigma=sigma, observed=log_y_data)

    return model


def remap_indices_for_prediction(
    df: pd.DataFrame,
    encoding: Encoding,
) -> dict[str, np.ndarray]:
    """
    Helper for posterior_predictive.py to convert a DataFrame of cells into
    the OOV-aware integer arrays the model expects.

    Input:
        df: rows with vessel_idx, berth_idx, service_idx columns (may contain -1).
        encoding: the same Encoding used at fit time.

    Output:
        Dict with keys 'vessel_idx', 'berth_idx', 'service_idx' and values
        that are int arrays with -1 replaced by n_levels (zero-offset slot).

    Description:
        Mirrors the same remapping done at fit time so that pm.set_data
        receives consistent indices at predict time.
    """
    # Using the SAME _remap_with_oov as at fit time guarantees predict-time
    # indices line up with the model's slots.
    return {
        "vessel_idx": _remap_with_oov(df["vessel_idx"].to_numpy(), encoding.n_vessel),
        "berth_idx": _remap_with_oov(df["berth_idx"].to_numpy(), encoding.n_berth),
        "service_idx": _remap_with_oov(df["service_idx"].to_numpy(), encoding.n_service),
    }

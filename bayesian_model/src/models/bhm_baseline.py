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

# `from __future__ import annotations` makes all type hints below behave as
# plain strings at runtime. The practical benefit: newer hint syntax like
# `dict[str, np.ndarray]` or `X | None` works even on older Python versions.
from __future__ import annotations

import numpy as np  # arrays / numeric helpers (np.where, np.zeros, ...)
import pandas as pd  # DataFrame = a table of rows and named columns
import pymc as pm  # PyMC: the Bayesian modelling library used here

# `..` means "go up one package level" — pull these from src/data_prep.py.
# OOV_INDEX is the sentinel (-1) used for "out-of-vocabulary" categories;
# Encoding is a dataclass holding how many levels each factor has.
from ..data_prep import OOV_INDEX, Encoding


# A leading underscore (`_remap_with_oov`) is a Python convention meaning
# "private helper" — intended for use inside this module only.
# `idx: np.ndarray` is a TYPE HINT: it documents that `idx` should be a NumPy
# array. `-> np.ndarray` documents the return type. Hints are not enforced.
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
    # np.where(condition, a, b): build a new array picking `a` where the
    # condition is True, else `b`. Here: wherever idx == -1 (OOV), use the
    # appended zero-offset slot at position n_levels; otherwise keep idx.
    # .astype("int64") forces integer dtype, since array indices must be ints.
    out = np.where(idx == OOV_INDEX, n_levels, idx).astype("int64")
    return out


# Arguments with `= value` are OPTIONAL with that default if the caller omits
# them. `**_unused` collects any EXTRA keyword arguments into a dict and throws
# them away — it lets every model in the registry accept the same call shape
# even if some priors don't apply to M0.
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
    # `coords` give human-readable labels to each dimension. PyMC attaches
    # these names to the posterior so plots/summaries show real category
    # names (e.g. a vessel type) instead of bare integer positions.
    coords = {
        "vessel": list(encoding.vessel.keys()),
        "berth": list(encoding.berth.keys()),
        "service": list(encoding.service.keys()),
    }

    # Pull each index column out as a NumPy array (.to_numpy()) and route any
    # OOV (-1) entries to the appended zero slot. n_vessel/etc. = number of
    # known levels, which is exactly the index of that extra zero row.
    v_idx = _remap_with_oov(train_df["vessel_idx"].to_numpy(), encoding.n_vessel)
    b_idx = _remap_with_oov(train_df["berth_idx"].to_numpy(), encoding.n_berth)
    s_idx = _remap_with_oov(train_df["service_idx"].to_numpy(), encoding.n_service)
    # The observed target: log of service time. Modelling on the log scale
    # turns multiplicative spread into additive Normal noise (a Lognormal y).
    log_y = train_df["log_service_time"].to_numpy()

    # `with pm.Model(...) as model:` opens a CONTEXT MANAGER. Every PyMC
    # random variable created inside this indented block is automatically
    # registered to `model`. When the block ends, the model is fully built.
    with pm.Model(coords=coords) as model:
        # pm.MutableData wraps an array so it can be SWAPPED later (via
        # pm.set_data) without recompiling the model — that is how
        # posterior_predictive.py reuses this model for new cells.
        # Mutable inputs so the same compiled model can be reused for
        # posterior predictive draws on arbitrary cells.
        vessel_idx_data = pm.MutableData("vessel_idx", v_idx)
        berth_idx_data = pm.MutableData("berth_idx", b_idx)
        service_idx_data = pm.MutableData("service_idx", s_idx)
        log_y_data = pm.MutableData("log_y", log_y)

        # PRIOR on the global intercept: a Normal random variable named
        # "alpha0" centred at alpha0_mean with spread alpha0_sd. This is our
        # belief about the average log(service_time) before seeing the data.
        # Global intercept on the log scale.
        alpha0 = pm.Normal("alpha0", mu=alpha0_mean, sigma=alpha0_sd)

        # pm.HalfNormal is a Normal folded to be >= 0 (good for a standard
        # deviation, which can't be negative). Each tau says how far a
        # factor's group effects are allowed to wander from 0 — small tau =>
        # strong shrinkage/pooling toward the global intercept.
        # Group-level standard deviations (pooling strength controls).
        tau_vessel = pm.HalfNormal("tau_vessel", sigma=tau_halfnormal_sd)
        tau_berth = pm.HalfNormal("tau_berth", sigma=tau_halfnormal_sd)
        tau_service = pm.HalfNormal("tau_service", sigma=tau_halfnormal_sd)

        # Non-centered parameterization: instead of sampling group effects
        # directly, sample standard normals z ~ Normal(0,1) (one per level via
        # dims=...) and scale by tau. This reshapes the posterior geometry so
        # NUTS does not get stuck in the "funnel" near tau ~ 0.
        z_vessel = pm.Normal("z_vessel", mu=0.0, sigma=1.0, dims="vessel")
        z_berth = pm.Normal("z_berth", mu=0.0, sigma=1.0, dims="berth")
        z_service = pm.Normal("z_service", mu=0.0, sigma=1.0, dims="service")

        # tau * z gives the actual per-level offset (the non-centered effect).
        # Group offsets, with a constant zero row appended at the end so
        # that OOV indices (remapped to n_levels) yield zero contribution.
        alpha_vessel_core = tau_vessel * z_vessel
        alpha_berth_core = tau_berth * z_berth
        alpha_service_core = tau_service * z_service

        # pm.Deterministic stores a derived quantity (not sampled directly,
        # computed from other variables) in the trace so we can inspect it.
        # pm.math.concatenate glues a constant 0.0 onto the end, creating the
        # extra slot that OOV rows (remapped to index n_levels) point at, so
        # an unknown category contributes a zero offset.
        alpha_vessel = pm.Deterministic(
            "alpha_vessel", pm.math.concatenate([alpha_vessel_core, [0.0]])
        )
        alpha_berth = pm.Deterministic(
            "alpha_berth", pm.math.concatenate([alpha_berth_core, [0.0]])
        )
        alpha_service = pm.Deterministic(
            "alpha_service", pm.math.concatenate([alpha_service_core, [0.0]])
        )

        # Linear predictor: the expected log(service_time) for each row.
        # `alpha_vessel[vessel_idx_data]` is fancy indexing — it gathers the
        # offset for each row's vessel, and likewise for berth/service.
        mu = (
            alpha0
            + alpha_vessel[vessel_idx_data]
            + alpha_berth[berth_idx_data]
            + alpha_service[service_idx_data]
        )

        # Residual spread of log(svc) around mu (>= 0, hence HalfNormal).
        # Residual log-scale standard deviation (Lognormal -> Normal on log y).
        sigma = pm.HalfNormal("sigma", sigma=sigma_halfnormal_sd)

        # LIKELIHOOD: passing `observed=` ties this Normal to the real data,
        # so sampling will fit mu/sigma to explain the observed log_y. This
        # is the only node connected to data; everything above are priors.
        pm.Normal("log_y_obs", mu=mu, sigma=sigma, observed=log_y_data)

    return model


# `-> dict[str, np.ndarray]` says this returns a dict whose keys are strings
# and whose values are NumPy arrays.
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
    # A dict literal {key: value, ...}. Using the SAME _remap_with_oov as at
    # fit time guarantees predict-time indices line up with the model's slots.
    return {
        "vessel_idx": _remap_with_oov(df["vessel_idx"].to_numpy(), encoding.n_vessel),
        "berth_idx": _remap_with_oov(df["berth_idx"].to_numpy(), encoding.n_berth),
        "service_idx": _remap_with_oov(df["service_idx"].to_numpy(), encoding.n_service),
    }

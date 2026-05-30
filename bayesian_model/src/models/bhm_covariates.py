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

# `from __future__ import annotations` makes every type hint in this file be
# stored as plain text instead of being evaluated. That lets us write modern
# hints like `int | None` on older Python versions without errors.
from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm  # PyMC: the Bayesian modelling library used to build/sample this model

# `from ..data_prep import ...` reaches up one package level (the `..`) to grab
# shared helpers/classes. OOV_INDEX is the sentinel (-1) for unseen categories.
from ..data_prep import OOV_INDEX, CovariateScaler, Encoding
# Reuse the baseline model's helper that reroutes OOV indices to the zero slot.
from .bhm_baseline import _remap_with_oov


# `def build_model(...) -> pm.Model:` defines a function. Each `name: type = value`
# is an argument with a type hint and a default value (used when the caller omits it).
# `**_unused` collects any extra keyword arguments into a dict named `_unused` and
# ignores them, so a shared caller can pass model-specific kwargs that M1 doesn't need.
# The `-> pm.Model` part says this function returns a PyMC Model object.
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
    # M1 cannot run without the covariate scaler, so fail loudly if it is missing.
    if scaler is None:
        raise ValueError("M1 requires a CovariateScaler. Call data_prep.prepare(with_covariates=True).")

    # `coords` names each model dimension and lists its labels. PyMC uses these
    # so posterior arrays come back labelled (e.g. one beta per feature) instead
    # of anonymous integers. `.keys()` gives the category strings in insertion order.
    coords = {
        "vessel": list(encoding.vessel.keys()),
        "berth": list(encoding.berth.keys()),
        "service": list(encoding.service.keys()),
        "feature": scaler.feature_cols,  # one beta coefficient per standardized covariate
    }

    # Pull each factor's integer codes out of the DataFrame as plain numpy arrays
    # and reroute any OOV (-1) codes to the appended zero-offset slot at n_levels.
    v_idx = _remap_with_oov(train_df["vessel_idx"].to_numpy(), encoding.n_vessel)
    b_idx = _remap_with_oov(train_df["berth_idx"].to_numpy(), encoding.n_berth)
    s_idx = _remap_with_oov(train_df["service_idx"].to_numpy(), encoding.n_service)
    log_y = train_df["log_service_time"].to_numpy()  # observed target on the log scale
    # Build the covariate matrix Z (rows = observations, cols = features).
    # `[f"z_{c}" for c in scaler.feature_cols]` is a list comprehension: it builds
    # the standardized column names (e.g. "z_log_trg") with an f-string, where the
    # `{c}` inside an f-string is replaced by the value of `c` each loop.
    Z = train_df[[f"z_{c}" for c in scaler.feature_cols]].to_numpy()

    # `with pm.Model(coords=coords) as model:` opens a context manager (the `with`
    # block). Every random variable created inside it is automatically registered
    # on `model`. When the block ends, the model holds the full graph.
    with pm.Model(coords=coords) as model:
        # pm.MutableData wraps the input arrays so the SAME compiled model can be
        # reused at predict time: posterior_predictive.py swaps in new rows via
        # pm.set_data without rebuilding the graph.
        vessel_idx_data = pm.MutableData("vessel_idx", v_idx)
        berth_idx_data = pm.MutableData("berth_idx", b_idx)
        service_idx_data = pm.MutableData("service_idx", s_idx)
        # Z is 2-D, so its dims are named: rows are observations ("obs", an
        # implicitly-sized dim) and columns line up with the "feature" coord.
        Z_data = pm.MutableData("Z", Z, dims=("obs", "feature"))
        log_y_data = pm.MutableData("log_y", log_y)

        # Global intercept on the log scale: a Normal prior centred at alpha0_mean.
        # pm.Normal("name", mu, sigma) declares a normally-distributed unknown the
        # sampler will estimate. The first argument is the variable's name in output.
        alpha0 = pm.Normal("alpha0", mu=alpha0_mean, sigma=alpha0_sd)
        # tau_* = how much each factor's group effects spread around the intercept.
        # pm.HalfNormal is a Normal folded to be >= 0, the natural prior for a sd.
        tau_vessel = pm.HalfNormal("tau_vessel", sigma=tau_halfnormal_sd)
        tau_berth = pm.HalfNormal("tau_berth", sigma=tau_halfnormal_sd)
        tau_service = pm.HalfNormal("tau_service", sigma=tau_halfnormal_sd)

        # Non-centered parameterization: draw standard normals then scale by tau.
        # `dims="vessel"` gives one z per vessel level. This re-parameterization
        # avoids the "funnel" geometry that makes NUTS sampling unreliable.
        z_vessel = pm.Normal("z_vessel", mu=0.0, sigma=1.0, dims="vessel")
        z_berth = pm.Normal("z_berth", mu=0.0, sigma=1.0, dims="berth")
        z_service = pm.Normal("z_service", mu=0.0, sigma=1.0, dims="service")

        # pm.Deterministic records a derived quantity (not sampled) so it is saved
        # in the trace. concatenate appends a constant 0.0 offset at index n_levels;
        # OOV rows were remapped to that slot, so they contribute zero group effect.
        alpha_vessel = pm.Deterministic("alpha_vessel", pm.math.concatenate([tau_vessel * z_vessel, [0.0]]))
        alpha_berth = pm.Deterministic("alpha_berth", pm.math.concatenate([tau_berth * z_berth, [0.0]]))
        alpha_service = pm.Deterministic("alpha_service", pm.math.concatenate([tau_service * z_service, [0.0]]))

        # One slope per standardized covariate. Because covariates are on the SD
        # scale, beta_sd is the prior "effect of a 1-SD change" on log(svc).
        beta = pm.Normal("beta", mu=0.0, sigma=beta_sd, dims="feature")

        # Linear predictor (expected log service time) for each row:
        # intercept + the three group offsets (indexed by this row's factor codes)
        # + the dot product of its covariates with their slopes (Z @ beta).
        mu = (
            alpha0
            + alpha_vessel[vessel_idx_data]
            + alpha_berth[berth_idx_data]
            + alpha_service[service_idx_data]
            + pm.math.dot(Z_data, beta)
        )
        # Residual spread on the log scale (this is a Lognormal model written as
        # a Normal on log y, so sigma is the log-scale standard deviation).
        sigma = pm.HalfNormal("sigma", sigma=sigma_halfnormal_sd)
        # The likelihood: observed=... ties this Normal to the real data, telling
        # the sampler to fit mu and sigma so log_y is plausible under the model.
        pm.Normal("log_y_obs", mu=mu, sigma=sigma, observed=log_y_data)

    return model

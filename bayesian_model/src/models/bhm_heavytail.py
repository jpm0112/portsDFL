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

# `from __future__ import annotations` makes all type hints (the `: type` and
# `-> type` annotations below) be treated as plain text, so newer syntax like
# `X | None` works even on older Python versions. Must be the first import.
from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm  # PyMC: the Bayesian modelling / MCMC library

# `..data_prep` means "go up one package level, then into data_prep".
# OOV_INDEX = sentinel (-1) for unseen categories; Encoding = the lookup tables.
from ..data_prep import OOV_INDEX, Encoding
# A leading underscore in `_remap_with_oov` is a convention meaning "internal /
# private helper". We reuse the baseline's OOV-routing helper here.
from .bhm_baseline import _remap_with_oov


# `def name(...) -> ReturnType:` defines a function; the part after `->` is the
# declared return type (here, a built PyMC model). Arguments written as
# `name: type = default` have both a type hint and a default value, so callers
# may omit them.
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
    # `**_unused` collects any extra keyword arguments into a dict and ignores
    # them; this lets every model share one uniform call signature.
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
    # A dict literal `{key: value, ...}`. `coords` names each model dimension
    # and lists its labels, so PyMC/ArviZ can tag outputs by category name.
    # `.keys()` yields the category strings; `list(...)` makes them a plain list.
    coords = {
        "vessel": list(encoding.vessel.keys()),
        "berth": list(encoding.berth.keys()),
        "service": list(encoding.service.keys()),
    }

    # Pull each factor's integer codes out of the DataFrame as a NumPy array,
    # then route any out-of-vocabulary (-1) code to the appended zero slot.
    v_idx = _remap_with_oov(train_df["vessel_idx"].to_numpy(), encoding.n_vessel)
    b_idx = _remap_with_oov(train_df["berth_idx"].to_numpy(), encoding.n_berth)
    s_idx = _remap_with_oov(train_df["service_idx"].to_numpy(), encoding.n_service)
    log_y = train_df["log_service_time"].to_numpy()  # the observed targets, log(svc)

    # `with pm.Model(...) as model:` opens a context manager: every random
    # variable created inside this indented block is automatically registered
    # on `model`. Leaving the block closes it. `as model` names the object.
    with pm.Model(coords=coords) as model:
        # pm.MutableData wraps an array as a named, swappable input. The same
        # compiled model can later be fed new indices via pm.set_data without
        # rebuilding it (used at posterior-predictive / prediction time).
        vessel_idx_data = pm.MutableData("vessel_idx", v_idx)
        berth_idx_data = pm.MutableData("berth_idx", b_idx)
        service_idx_data = pm.MutableData("service_idx", s_idx)
        log_y_data = pm.MutableData("log_y", log_y)

        # A prior: `pm.Normal("name", mu, sigma)` declares a random variable
        # with a Normal(mu, sigma) prior distribution. Here the global
        # intercept of log(service time).
        alpha0 = pm.Normal("alpha0", mu=alpha0_mean, sigma=alpha0_sd)
        # pm.HalfNormal is a Normal folded to be >= 0 (good for scale params).
        # These tau's control how strongly each group shrinks toward alpha0.
        tau_vessel = pm.HalfNormal("tau_vessel", sigma=tau_halfnormal_sd)
        tau_berth = pm.HalfNormal("tau_berth", sigma=tau_halfnormal_sd)
        tau_service = pm.HalfNormal("tau_service", sigma=tau_halfnormal_sd)

        # Non-centered parameterization: raw standard-normal offsets, one per
        # category (`dims="vessel"` gives this variable the vessel dimension).
        z_vessel = pm.Normal("z_vessel", mu=0.0, sigma=1.0, dims="vessel")
        z_berth = pm.Normal("z_berth", mu=0.0, sigma=1.0, dims="berth")
        z_service = pm.Normal("z_service", mu=0.0, sigma=1.0, dims="service")

        # pm.Deterministic records a derived quantity (not sampled directly) so
        # it is saved in the trace. Each group offset = tau * z, with a 0.0
        # appended at the end as the OOV slot (so OOV rows add zero offset).
        alpha_vessel = pm.Deterministic("alpha_vessel", pm.math.concatenate([tau_vessel * z_vessel, [0.0]]))
        alpha_berth = pm.Deterministic("alpha_berth", pm.math.concatenate([tau_berth * z_berth, [0.0]]))
        alpha_service = pm.Deterministic("alpha_service", pm.math.concatenate([tau_service * z_service, [0.0]]))

        # Linear predictor (the location mu of each observation). Indexing an
        # offset array by the integer-code array picks each row's group effect.
        mu = (
            alpha0
            + alpha_vessel[vessel_idx_data]
            + alpha_berth[berth_idx_data]
            + alpha_service[service_idx_data]
        )
        sigma = pm.HalfNormal("sigma", sigma=sigma_halfnormal_sd)  # scale of the noise

        # Reparameterize nu so it stays > 1 (variance defined for nu>2).
        # pm.Gamma(alpha, beta) here has mean alpha/beta = 2/0.1 = 20, so the
        # degrees-of-freedom nu has prior mean ~21 (near-Normal but with room
        # for heavy tails). nu controls tail weight: small nu -> heavier tails.
        nu_minus_one = pm.Gamma("nu_minus_one", alpha=nu_alpha, beta=nu_beta)
        nu = pm.Deterministic("nu", nu_minus_one + 1.0)

        # The likelihood: `observed=` ties this StudentT distribution to the
        # real data, so sampling infers the parameters that best explain it.
        # StudentT (vs Normal in M0) lets occasional large residuals occur
        # without inflating sigma -- the heavy-tail fix this model is about.
        pm.StudentT("log_y_obs", nu=nu, mu=mu, sigma=sigma, observed=log_y_data)

    # The fully-specified model is returned; sampling happens elsewhere (fit.py
    # calls pm.sample for MCMC posterior draws).
    return model

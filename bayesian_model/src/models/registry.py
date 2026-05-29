"""
Model registry mapping model_key strings to their build functions and
unified setter for predictive MutableData.

Every entry returns a pm.Model when called with:
    builder(train_df, encoding, scaler, **priors_kwargs) -> pm.Model

set_predict_data(model, df, encoding, scaler) wires the right inputs into
whichever MutableData fields the model exposes (handling M1's Z matrix
and M4's interaction index automatically).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm

from ..data_prep import Encoding
from .bhm_baseline import build_model as build_m0
from .bhm_baseline import _remap_with_oov
from .bhm_covariates import build_model as build_m1
from .bhm_heavytail import build_model as build_m2
from .bhm_heteroscedastic import build_model as build_m3
from .bhm_interactions import build_model as build_m4


MODEL_REGISTRY = {
    "m0_baseline": build_m0,
    "m1_covariates": build_m1,
    "m2_heavytail": build_m2,
    "m3_heteroscedastic": build_m3,
    "m4_interactions": build_m4,
}


def build(model_key: str, train_df, encoding, scaler, **priors_kwargs):
    """
    Dispatch to the registered builder for `model_key`.

    Input: model_key plus standard build args.
    Output: pm.Model.
    """
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_key '{model_key}'. Registered: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_key](train_df=train_df, encoding=encoding, scaler=scaler, **priors_kwargs)


def set_predict_data(model: pm.Model, df: pd.DataFrame, encoding: Encoding, scaler) -> None:
    """
    Wire prediction inputs into whichever MutableData fields the model exposes.

    Input:
        model: the rebuilt pm.Model.
        df: rows to predict for; must have vessel_idx/berth_idx/service_idx;
            for M1 must also have z_* columns; for M4 must have vessel_idx
            and berth_idx (used to compute vb_idx via the model's stored
            interaction selection).
        encoding: same Encoding used at fit time.
        scaler: CovariateScaler used at fit time (None for non-covariate models).

    Output: none (mutates the model's MutableData in place).

    Description:
        The model is the source of truth for which fields exist; we only
        set what we find. log_y is set to zeros at predict time (its values
        are ignored by sample_posterior_predictive, but the shape must match).
    """
    n = len(df)
    data = {
        "vessel_idx": _remap_with_oov(df["vessel_idx"].to_numpy(), encoding.n_vessel),
        "berth_idx": _remap_with_oov(df["berth_idx"].to_numpy(), encoding.n_berth),
        "service_idx": _remap_with_oov(df["service_idx"].to_numpy(), encoding.n_service),
        "log_y": np.zeros(n, dtype=float),
    }
    if "Z" in model.named_vars and scaler is not None:
        data["Z"] = df[[f"z_{c}" for c in scaler.feature_cols]].to_numpy()
    if "vb_idx" in model.named_vars:
        vb_index = getattr(model, "_vb_index", {})
        n_sel = len(vb_index)
        data["vb_idx"] = np.array(
            [vb_index.get((int(v), int(b)), n_sel) for v, b in zip(df["vessel_idx"], df["berth_idx"])],
            dtype="int64",
        )
    with model:
        pm.set_data(data)

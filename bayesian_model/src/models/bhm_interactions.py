"""
M4: M0 + selective vessel * berth interaction terms.

Adds an interaction offset gamma[(v, b)] for each (vessel, berth) cell with
n_train >= min_n. Cells below the threshold get a zero interaction (model
falls back to the additive structure of M0 there). This avoids inventing
structure where the data is too sparse to estimate it.

Likelihood / hierarchy:
    log(svc_i) ~ Normal(mu_i, sigma)
    mu_i = alpha0 + alpha_vessel + alpha_berth + alpha_service + gamma_vb[i]
    gamma_vb[k]    = tau_vb * z_vb[k]      (non-centered, k indexes selected cells)
    tau_vb         ~ HalfNormal(tau_halfnormal_sd)
    z_vb[k]        ~ Normal(0, 1)

Selected cells are determined at build time from train_df. A row that
maps to an unselected (or unseen) (v, b) gets routed to the appended
zero slot — same OOV trick as M0.
"""

# See bhm_covariates.py: this keeps type hints as un-evaluated text.
from __future__ import annotations

# Dict and Tuple are type hints. `Dict[Tuple[int, int], int]` reads as
# "a dict whose keys are (int, int) pairs and whose values are ints".
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pymc as pm  # PyMC: builds the Bayesian model and runs MCMC sampling

from ..data_prep import OOV_INDEX, Encoding
from .bhm_baseline import _remap_with_oov  # reuse OOV->zero-slot remapping helper


# Default minimum cell size to estimate an interaction. Below this, the
# (vessel, berth) cell falls back to the additive structure.
# A module-level constant (ALL_CAPS by convention = "treat as fixed").
DEFAULT_MIN_INTERACTION_N = 30


def _select_interaction_cells(
    train_df: pd.DataFrame, min_n: int
) -> Tuple[Dict[Tuple[int, int], int], np.ndarray]:
    """
    Identify (vessel, berth) cells with enough data and assign them indices.

    Input:
        train_df: training rows with vessel_idx, berth_idx columns.
        min_n: minimum cell count to fit an interaction.

    Output:
        (vb_index, vb_idx_train) where:
            vb_index: dict (vessel_idx, berth_idx) -> contiguous int index
                      for each *selected* cell. Cells not in this dict have
                      no interaction term and route to the zero slot.
            vb_idx_train: int array of length len(train_df) giving the
                          interaction index for each training row, with
                          the zero-slot index n_selected used for non-
                          selected cells.

    Description:
        Centralized so that build_model and remap_indices_for_prediction
        agree on which cells got interactions.
    """
    # Count how many training rows fall in each (vessel, berth) combination.
    # groupby(...).size() returns a Series indexed by the (vessel_idx, berth_idx) pair.
    counts = train_df.groupby(["vessel_idx", "berth_idx"]).size()
    # Keep only cells with at least min_n rows. `[k for k, n in counts.items() if ...]`
    # is a list comprehension iterating over (cell, count) pairs. sorted() makes the
    # order deterministic so indices are reproducible across runs.
    selected = sorted([k for k, n in counts.items() if n >= min_n])
    # Map each selected cell to a contiguous index 0,1,2,... `enumerate` yields
    # (position, item) pairs; this dict comprehension flips them into cell -> index.
    vb_index = {cell: i for i, cell in enumerate(selected)}
    n_sel = len(selected)  # the zero-slot lives at index n_sel (one past the last cell)
    # For every training row, look up its interaction index; rows whose cell was
    # not selected fall back to the zero slot n_sel via dict.get(key, default).
    # `zip(...)` walks the two columns together; int() guards against numpy int types
    # so the tuple keys match those built from counts.items().
    vb_idx_train = np.array(
        [vb_index.get((int(v), int(b)), n_sel) for v, b in zip(train_df["vessel_idx"], train_df["berth_idx"])],
        dtype="int64",
    )
    return vb_index, vb_idx_train


def remap_interaction_for_prediction(
    df: pd.DataFrame, vb_index: Dict[Tuple[int, int], int]
) -> np.ndarray:
    """
    Map a DataFrame's rows to the interaction-cell indices used at fit time.

    Input/Output: DataFrame with vessel_idx/berth_idx columns -> int array.
    Description: rows not in vb_index map to the zero slot (n_selected).
    """
    # n_sel = number of selected cells; it doubles as the zero-slot index, so any
    # cell missing from vb_index (unselected at fit time, or unseen) routes there.
    # This MUST mirror the indexing used in _select_interaction_cells so fit-time
    # and predict-time interaction indices line up against the same gamma_vb vector.
    n_sel = len(vb_index)
    return np.array(
        [vb_index.get((int(v), int(b)), n_sel) for v, b in zip(df["vessel_idx"], df["berth_idx"])],
        dtype="int64",
    )


def build_model(
    train_df: pd.DataFrame,
    encoding: Encoding,
    scaler=None,
    alpha0_mean: float = 3.47,
    alpha0_sd: float = 1.0,
    tau_halfnormal_sd: float = 0.5,
    sigma_halfnormal_sd: float = 0.7,
    min_interaction_n: int = DEFAULT_MIN_INTERACTION_N,
    **_unused,
) -> pm.Model:
    """
    Build M4: M0 + selective vessel * berth interactions.

    Input:
        Same as M0, plus min_interaction_n for the cell-size threshold.

    Output:
        pm.Model. Also stashes the vb_index dict on model.named_vars via
        a Deterministic so downstream code can recover which cells were
        selected (otherwise we'd need to repeat the selection logic).

    Description:
        We avoid repeating cell selection at predict time by attaching
        vb_index to the model object. Since pm.Model is a Python object,
        we just store it as an attribute (`model._vb_index`). PyMC does
        not serialize this automatically; downstream code that rebuilds
        the model uses the same min_interaction_n and the same training
        data, so the selection is reproducible.
    """
    coords = {
        "vessel": list(encoding.vessel.keys()),
        "berth": list(encoding.berth.keys()),
        "service": list(encoding.service.keys()),
    }

    # Remap each factor's OOV (-1) codes to the appended zero-offset slot (see baseline).
    v_idx = _remap_with_oov(train_df["vessel_idx"].to_numpy(), encoding.n_vessel)
    b_idx = _remap_with_oov(train_df["berth_idx"].to_numpy(), encoding.n_berth)
    s_idx = _remap_with_oov(train_df["service_idx"].to_numpy(), encoding.n_service)
    log_y = train_df["log_service_time"].to_numpy()

    # Decide which (vessel, berth) cells have enough data to get their own
    # interaction offset, and the per-row index into that set of cells.
    vb_index, vb_idx_train = _select_interaction_cells(train_df, min_n=min_interaction_n)
    n_vb = len(vb_index)  # number of interaction cells (excludes the zero slot)
    # Human-readable label per interaction cell, e.g. "v3_b7", for the trace.
    # `for (v, b) in vb_index` iterates the dict's keys, which are (vessel, berth) tuples.
    coords["vb_cell"] = [f"v{v}_b{b}" for (v, b) in vb_index]

    # `with pm.Model(...) as model:` collects every variable defined inside as part
    # of this model's graph (see bhm_covariates.py for the full explanation).
    with pm.Model(coords=coords) as model:
        # Mutable inputs so the compiled model can be reused for predictions.
        vessel_idx_data = pm.MutableData("vessel_idx", v_idx)
        berth_idx_data = pm.MutableData("berth_idx", b_idx)
        service_idx_data = pm.MutableData("service_idx", s_idx)
        vb_idx_data = pm.MutableData("vb_idx", vb_idx_train)  # per-row interaction-cell index
        log_y_data = pm.MutableData("log_y", log_y)

        # Global intercept prior on the log scale.
        alpha0 = pm.Normal("alpha0", mu=alpha0_mean, sigma=alpha0_sd)
        # HalfNormal (>= 0) priors on the spread of each group's effects.
        tau_vessel = pm.HalfNormal("tau_vessel", sigma=tau_halfnormal_sd)
        tau_berth = pm.HalfNormal("tau_berth", sigma=tau_halfnormal_sd)
        tau_service = pm.HalfNormal("tau_service", sigma=tau_halfnormal_sd)
        tau_vb = pm.HalfNormal("tau_vb", sigma=tau_halfnormal_sd)  # spread of interaction offsets

        # Non-centered standard normals, one per level, scaled by tau later.
        z_vessel = pm.Normal("z_vessel", mu=0.0, sigma=1.0, dims="vessel")
        z_berth = pm.Normal("z_berth", mu=0.0, sigma=1.0, dims="berth")
        z_service = pm.Normal("z_service", mu=0.0, sigma=1.0, dims="service")
        if n_vb > 0:
            # One interaction offset per selected cell, non-centered like the rest.
            z_vb = pm.Normal("z_vb", mu=0.0, sigma=1.0, dims="vb_cell")
            gamma_vb_core = tau_vb * z_vb
        else:
            # No cells passed threshold; degenerate to M0 with a vestigial tau_vb.
            # pm.math.zeros((0,)) is an empty length-0 vector; the 0.0 appended
            # below then becomes the single (zero) slot every row maps to.
            gamma_vb_core = pm.math.zeros((0,))

        # Group offsets with a trailing constant 0.0 so OOV rows contribute nothing.
        alpha_vessel = pm.Deterministic("alpha_vessel", pm.math.concatenate([tau_vessel * z_vessel, [0.0]]))
        alpha_berth = pm.Deterministic("alpha_berth", pm.math.concatenate([tau_berth * z_berth, [0.0]]))
        alpha_service = pm.Deterministic("alpha_service", pm.math.concatenate([tau_service * z_service, [0.0]]))
        # Interaction offsets, also with a trailing 0.0: rows in unselected/unseen
        # cells were routed to that final slot and so get a zero interaction.
        gamma_vb = pm.Deterministic("gamma_vb", pm.math.concatenate([gamma_vb_core, [0.0]]))

        # Expected log service time = intercept + the three additive group offsets
        # (indexed by each row's factor codes) + the (vessel, berth) interaction.
        mu = (
            alpha0
            + alpha_vessel[vessel_idx_data]
            + alpha_berth[berth_idx_data]
            + alpha_service[service_idx_data]
            + gamma_vb[vb_idx_data]
        )
        # Residual log-scale spread (Lognormal model expressed as Normal on log y).
        sigma = pm.HalfNormal("sigma", sigma=sigma_halfnormal_sd)
        # Likelihood: observed=... attaches the real data so sampling fits mu/sigma.
        pm.Normal("log_y_obs", mu=mu, sigma=sigma, observed=log_y_data)

    # Stash the selection on the Python object so predict-time code can rebuild the
    # exact same cell->index mapping. `# type: ignore[...]` silences a type checker
    # complaining about setting an attribute PyMC doesn't formally declare.
    model._vb_index = vb_index  # type: ignore[attr-defined]
    model._min_interaction_n = min_interaction_n  # type: ignore[attr-defined]
    return model

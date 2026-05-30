"""Week-long DBAP instance builder (the pre-solve slicing step).

This module is the **pre-processing step that runs before the MILP**. It takes
the historical/forward vessel-call data, slices a single 7-day planning window,
and emits ONE self-contained ``BAPInstance`` (horizon = 168 h) for the
optimizer. The MILP itself solves exactly that one week — there is no
multi-week, rolling-horizon, or cross-week carry-over logic here or downstream.

Two entry points:

- ``build_weekly_instance`` — slice a real week out of a dataframe / CSV. It
  reads the build-script schema directly (``F. arribo`` / ``Fecha arribo``,
  ``Terminal`` / ``Sitio``, ``vessel_type_group`` / ``Tipo nave (agrupado)``,
  target ``estadia_sitio_hours``) and does NOT depend on the (separately
  missing) ``ports_dfl.data`` loader subpackage.
- ``generate_synthetic_weekly_instance`` — a no-data fallback that fabricates a
  realistic week (arrivals, types, service times, services, compatibility) so
  the planner and tests run without the proprietary port data.

Both return a :class:`WeeklyInstanceBundle` carrying the ``BAPInstance`` plus
the per-vessel metadata the planner needs for tables and Gantt charts. The
bundle is numpy-only (no pandas object retained), so the synthetic path imports
without pandas.

Arrival basis: ``aᵢ`` is the arrival timestamp relative to the week start, in
hours (default column ``F. arribo`` — anchorage arrival — so wait = sᵢ − aᵢ is
the anchorage-to-berth wait that service vessels must avoid). See the decisions
log, Q8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .berths import DEFAULT_BERTHS, VESSEL_TYPE_GROUPS, Berth, derive_berths_from_history, vessel_berth_compat
from .instance import BAPInstance

# Column-name aliases across the two dataset schemas (engineered
# training_dataset.csv vs source-faithful clean_dataset.csv).
_ARRIVAL_ALIASES = ("F. arribo", "Fecha arribo")
_TYPE_ALIASES = ("vessel_type_group", "Tipo nave (agrupado)")
_BERTH_ALIASES = ("Terminal", "Sitio")
_ID_ALIASES = ("Cód. nave", "Código nave")
_NAME_ALIASES = ("Nave",)
_TAU_ALIASES = ("estadia_sitio_hours",)

# Rough by-type median berth-occupation hours (from the project's data
# summaries) — used only by the synthetic generator.
_TYPE_TAU_MEDIAN = {
    "Container": 36.0,
    "Dry Bulk": 74.0,
    "Vehicle Carrier": 36.0,
    "Liquid Bulk": 22.0,
    "General Cargo": 36.0,
    "Passenger": 19.0,
    "Other": 65.0,
}
# Approximate share of calls by type (Container-dominated port).
_TYPE_MIX = {
    "Container": 0.55,
    "Dry Bulk": 0.17,
    "Vehicle Carrier": 0.13,
    "Liquid Bulk": 0.10,
    "General Cargo": 0.03,
    "Passenger": 0.015,
    "Other": 0.005,
}


@dataclass(frozen=True)
class WeeklyInstanceBundle:
    """A built weekly instance plus the metadata needed to report a schedule.

    All per-vessel arrays are length N and index-aligned with the
    ``BAPInstance`` (vessel i is row i everywhere).
    """

    instance: BAPInstance
    berths: list[Berth]
    vessel_ids: list
    vessel_names: list
    vessel_types: list[str]
    arrivals_h: np.ndarray          # (N,) arrival relative to week start, hours
    tau_h: np.ndarray               # (N,) service time used in this instance
    weights: np.ndarray             # (N,) priority weights
    is_service: np.ndarray          # (N,) bool
    latest_start_h: np.ndarray      # (N,) latest start; inf for non-service
    week_start: str
    week_end: str
    source: str                     # provenance string

    @property
    def n_vessels(self) -> int:
        return self.instance.n_vessels

    @property
    def n_berths(self) -> int:
        return self.instance.n_berths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick(df, aliases, role: str, required: bool = True):
    """Return the first column in ``df`` matching ``aliases`` (or None)."""
    for name in aliases:
        if name in df.columns:
            return name
    if required:
        raise KeyError(
            f"No {role} column found; tried {list(aliases)}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def _resolve_service_mask(selector, df_week, id_col) -> np.ndarray:
    """Resolve a service selector to a length-N boolean mask.

    ``selector`` may be: None (no services); a callable(df_week)->bool array;
    a boolean array of length N; or a collection of vessel ids.
    """
    n = len(df_week)
    if selector is None:
        return np.zeros(n, dtype=bool)
    if callable(selector):
        return np.asarray(selector(df_week), dtype=bool).reshape(n)
    sel = list(selector)
    is_bool_like = len(sel) == n and all(
        isinstance(v, (bool, np.bool_)) for v in sel
    )
    if is_bool_like:
        return np.asarray(sel, dtype=bool)
    if id_col is None:
        raise ValueError(
            "service_selector is a collection of ids but no id column was found."
        )
    wanted = {str(v) for v in sel}
    ids = df_week[id_col].astype(str).tolist()
    return np.array([vid in wanted for vid in ids], dtype=bool)


def _build_bundle(
    *,
    vessel_ids,
    vessel_names,
    vessel_types,
    arrivals_h,
    tau_h,
    weights,
    is_service,
    latest_start_h,
    berths,
    week_start,
    week_end,
    source,
    big_m,
) -> WeeklyInstanceBundle:
    """Assemble a BAPInstance + bundle from already-computed per-vessel arrays."""
    arrivals_h = np.asarray(arrivals_h, dtype=np.float32)
    tau_h = np.asarray(tau_h, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    is_service = np.asarray(is_service, dtype=bool)
    latest_start_h = np.asarray(latest_start_h, dtype=np.float32)
    n = len(arrivals_h)

    compat = vessel_berth_compat(vessel_types, berths, allow_unmatched=False)

    if big_m is None:
        horizon = float(np.max(arrivals_h, initial=0.0))
        big_m = float(horizon + float(tau_h.sum()) + 1.0)

    # Only pass window arrays when at least one service vessel exists.
    if bool(is_service.any()):
        service_arr = is_service
        latest_arr = latest_start_h
    else:
        service_arr = None
        latest_arr = None

    instance = BAPInstance(
        n_vessels=n,
        n_berths=len(berths),
        arrivals=arrivals_h,
        weights=weights,
        big_m=big_m,
        latest_start=latest_arr,
        berth_compat=compat,
        service=service_arr,
    )
    return WeeklyInstanceBundle(
        instance=instance,
        berths=list(berths),
        vessel_ids=list(vessel_ids),
        vessel_names=list(vessel_names),
        vessel_types=[str(t) for t in vessel_types],
        arrivals_h=arrivals_h,
        tau_h=tau_h,
        weights=weights,
        is_service=is_service,
        latest_start_h=latest_start_h,
        week_start=str(week_start),
        week_end=str(week_end),
        source=source,
    )


# ---------------------------------------------------------------------------
# Real-data builder
# ---------------------------------------------------------------------------

def build_weekly_instance(
    source,
    week_start,
    *,
    week_days: int = 7,
    berth_col: str | None = None,
    type_col: str | None = None,
    arrival_col: str | None = None,
    id_col: str | None = None,
    name_col: str | None = None,
    tau_col: str | None = None,
    tau=None,
    weights=None,
    base_weight: float = 1.0,
    service_weight: float = 3.0,
    service_selector=None,
    service_slack_hours: float = 0.0,
    berths: list[Berth] | None = None,
    min_compat_count: int = 1,
    big_m: float | None = None,
) -> WeeklyInstanceBundle:
    """Slice one 7-day window out of the call data and build a BAPInstance.

    Args:
        source: a pandas DataFrame, or a path to a CSV (build-script schema).
        week_start: start of the planning week (anything ``pd.to_datetime``
            accepts, e.g. ``"2025-03-03"``).
        week_days: window length in days (default 7).
        berth_col/type_col/arrival_col/id_col/name_col/tau_col: column names;
            auto-detected from the known aliases when ``None``.
        tau: service-time override — an array (length = #vessels in the week)
            or a callable ``df_week -> array``. Default uses the true
            ``estadia_sitio_hours`` column (predict-then-optimize / DFL can pass
            model predictions here instead).
        weights: priority-weight override (array or callable). Default assigns
            ``base_weight`` to ordinary vessels and ``service_weight`` to
            service vessels.
        service_selector: which vessels are priority "services" — None, a bool
            array, a callable, or a collection of vessel ids. See
            ``_resolve_service_mask``.
        service_slack_hours: latest-start slack lᵣ = aᵣ + slack (0 ⇒ no waiting).
        berths: berth catalog; when ``None`` it is derived from the FULL input
            via ``derive_berths_from_history`` (so rare types in the chosen week
            still get the right compatibility).
        min_compat_count: threshold passed to ``derive_berths_from_history``.
        big_m: precedence big-M; auto-sized when ``None``.

    Returns:
        A :class:`WeeklyInstanceBundle`.

    Raises:
        ValueError: if no vessel calls fall inside the window.
    """
    import pandas as pd  # local import: synthetic path needs no pandas

    if isinstance(source, str):
        df = pd.read_csv(source)
        provenance = f"csv:{source}"
    else:
        df = source.copy()
        provenance = "dataframe"

    arrival_col = arrival_col or _pick(df, _ARRIVAL_ALIASES, "arrival")
    type_col = type_col or _pick(df, _TYPE_ALIASES, "vessel_type")
    berth_col = berth_col or _pick(df, _BERTH_ALIASES, "berth")
    id_col = id_col or _pick(df, _ID_ALIASES, "vessel_id", required=False)
    name_col = name_col or _pick(df, _NAME_ALIASES, "vessel_name", required=False)
    tau_col = tau_col or _pick(df, _TAU_ALIASES, "service_time", required=(tau is None))

    # Derive the berth catalog from the full history (before slicing).
    if berths is None:
        berths = derive_berths_from_history(
            df, berth_col=berth_col, type_col=type_col, min_count=min_compat_count
        )

    arr = pd.to_datetime(df[arrival_col])
    ws = pd.to_datetime(week_start)
    we = ws + pd.Timedelta(days=week_days)
    mask = (arr >= ws) & (arr < we)
    df_week = df.loc[mask].copy()
    if len(df_week) == 0:
        raise ValueError(
            f"No vessel calls with {arrival_col} in [{ws.date()}, {we.date()}). "
            f"Pick a week within the data's coverage."
        )
    df_week = df_week.sort_values(arrival_col).reset_index(drop=True)
    arr_week = pd.to_datetime(df_week[arrival_col])

    arrivals_h = ((arr_week - ws).dt.total_seconds() / 3600.0).to_numpy(dtype=np.float32)
    arrivals_h = np.maximum(arrivals_h, 0.0)  # guard tiny negatives from rounding

    vessel_types = df_week[type_col].astype(str).tolist()
    vessel_ids = (
        df_week[id_col].tolist() if id_col else list(range(len(df_week)))
    )
    vessel_names = (
        df_week[name_col].astype(str).tolist() if name_col else [str(v) for v in vessel_ids]
    )

    # Service times τ.
    if tau is None:
        tau_h = df_week[tau_col].to_numpy(dtype=np.float32)
    elif callable(tau):
        tau_h = np.asarray(tau(df_week), dtype=np.float32)
    else:
        tau_h = np.asarray(tau, dtype=np.float32)
    if tau_h.shape != (len(df_week),):
        raise ValueError(f"tau has shape {tau_h.shape}, expected {(len(df_week),)}")

    is_service = _resolve_service_mask(service_selector, df_week, id_col)

    # Weights.
    if weights is None:
        w = np.full(len(df_week), float(base_weight), dtype=np.float32)
        w[is_service] = float(service_weight)
    elif callable(weights):
        w = np.asarray(weights(df_week), dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32)

    # Latest-start windows (inf where no window).
    latest_start_h = np.full(len(df_week), np.inf, dtype=np.float32)
    latest_start_h[is_service] = arrivals_h[is_service] + float(service_slack_hours)

    return _build_bundle(
        vessel_ids=vessel_ids,
        vessel_names=vessel_names,
        vessel_types=vessel_types,
        arrivals_h=arrivals_h,
        tau_h=tau_h,
        weights=w,
        is_service=is_service,
        latest_start_h=latest_start_h,
        berths=berths,
        week_start=str(pd.to_datetime(week_start).date()),
        week_end=str((ws + pd.Timedelta(days=week_days)).date()),
        source=provenance,
        big_m=big_m,
    )


# ---------------------------------------------------------------------------
# Synthetic no-data builder
# ---------------------------------------------------------------------------

def generate_synthetic_weekly_instance(
    n_vessels: int = 18,
    *,
    week_days: int = 7,
    seed: int = 0,
    berths: list[Berth] | tuple[Berth, ...] = DEFAULT_BERTHS,
    n_services: int = 2,
    service_slack_hours: float = 0.0,
    base_weight: float = 1.0,
    service_weight: float = 3.0,
) -> WeeklyInstanceBundle:
    """Fabricate a realistic week (no data needed) for demos and tests.

    Vessel types are sampled from the port's rough mix; service times from the
    by-type medians with lognormal noise; arrivals spread across the week. The
    service vessels are chosen among Container calls (which can use ≥2 berths in
    ``DEFAULT_BERTHS``), so the hard-window instance is feasible even at
    ``service_slack_hours=0``.
    """
    rng = np.random.default_rng(seed)
    berths = list(berths)

    types_pool = list(_TYPE_MIX.keys())
    probs = np.array([_TYPE_MIX[t] for t in types_pool], dtype=float)
    probs = probs / probs.sum()
    vessel_types = list(rng.choice(types_pool, size=n_vessels, p=probs))

    horizon = week_days * 24.0
    arrivals_h = np.sort(rng.uniform(0.0, 0.9 * horizon, size=n_vessels)).astype(np.float32)

    tau_h = np.array(
        [_TYPE_TAU_MEDIAN[t] * float(rng.lognormal(0.0, 0.3)) for t in vessel_types],
        dtype=np.float32,
    )

    # Services: prefer Container vessels (≥2 compatible berths), most-separated.
    container_idx = [i for i, t in enumerate(vessel_types) if t == "Container"]
    chosen = container_idx[:n_services] if container_idx else list(range(n_services))
    is_service = np.zeros(n_vessels, dtype=bool)
    is_service[chosen] = True

    weights = np.full(n_vessels, float(base_weight), dtype=np.float32)
    weights[is_service] = float(service_weight)

    latest_start_h = np.full(n_vessels, np.inf, dtype=np.float32)
    latest_start_h[is_service] = arrivals_h[is_service] + float(service_slack_hours)

    vessel_ids = [f"V{i:02d}" for i in range(n_vessels)]
    vessel_names = [f"synthetic-{i:02d}" for i in range(n_vessels)]

    return _build_bundle(
        vessel_ids=vessel_ids,
        vessel_names=vessel_names,
        vessel_types=vessel_types,
        arrivals_h=arrivals_h,
        tau_h=tau_h,
        weights=weights,
        is_service=is_service,
        latest_start_h=latest_start_h,
        berths=berths,
        week_start="synthetic-week-0",
        week_end=f"synthetic-week-{week_days}d",
        source=f"synthetic(seed={seed},n={n_vessels})",
        big_m=None,
    )

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

# `from __future__ import annotations` makes every type hint in this file a
# plain string at runtime instead of being evaluated. That lets us write hints
# like `list[Berth] | None` even on older Pythons, and avoids import-order issues.
from __future__ import annotations

# `dataclass` is a decorator (see its use below) that auto-writes boilerplate
# like __init__ for a class whose job is just to hold fields.
from dataclasses import dataclass

import numpy as np

# Relative imports (the leading dot means "from this same package"). We pull in
# the berth catalog + the compatibility-matrix builder, and the BAPInstance
# descriptor that the MILP solver consumes.
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


# `@dataclass(frozen=True)` is a decorator applied to the class right below it.
# It auto-generates __init__/__repr__/__eq__ from the fields declared as class
# attributes. `frozen=True` makes instances read-only after creation (assigning
# to a field raises), which is handy for an immutable "result bundle".
@dataclass(frozen=True)
class WeeklyInstanceBundle:
    """A built weekly instance plus the metadata needed to report a schedule.

    All per-vessel arrays are length N and index-aligned with the
    ``BAPInstance`` (vessel i is row i everywhere).
    """

    # Each line below is a FIELD with a type hint (`name: type`). The dataclass
    # turns these into constructor arguments, in this order.
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

    # `@property` lets you call this like an attribute (`bundle.n_vessels`) with
    # no parentheses, even though it runs a method. `self` is the instance, and
    # `-> int` says it returns an int. We just forward to the inner instance so
    # the count lives in one place.
    @property
    def n_vessels(self) -> int:
        return self.instance.n_vessels

    @property
    def n_berths(self) -> int:
        return self.instance.n_berths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A leading underscore (`_pick`) is a convention meaning "private helper" —
# not part of the module's public API. `role: str` and `required: bool = True`
# are type-hinted parameters; `= True` gives `required` a default value.
def _pick(df, aliases, role: str, required: bool = True):
    """Return the first column in ``df`` matching ``aliases`` (or None)."""
    # Walk the candidate names in priority order; return the first one the
    # dataframe actually has. `df.columns` is the list of column labels.
    for name in aliases:
        if name in df.columns:
            return name
    if required:
        # An f-string (the `f"..."` prefix) lets you embed expressions in
        # `{ }` directly inside the string for a clear error message.
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
    # Case 1: no selector -> nobody is a service. All-False mask.
    if selector is None:
        return np.zeros(n, dtype=bool)
    # Case 2: a function. `callable(x)` is True if x can be called like `x(...)`.
    # We call it on the week's dataframe and coerce the result to a bool array;
    # `.reshape(n)` forces length N (raising loudly if the callable returns a
    # wrong-length result).
    if callable(selector):
        return np.asarray(selector(df_week), dtype=bool).reshape(n)
    # Otherwise materialise the selector into a list so we can inspect it.
    sel = list(selector)
    # Case 3 detection: is this already a length-N boolean MASK? It qualifies
    # only if every element is a Python/numpy bool. `all(... for ... in ...)`
    # is a generator expression: True iff the test holds for every element.
    is_bool_like = len(sel) == n and all(
        isinstance(v, (bool, np.bool_)) for v in sel
    )
    if is_bool_like:
        return np.asarray(sel, dtype=bool)
    # Case 4: treat the selector as a collection of vessel IDs. We need an id
    # column to match against.
    if id_col is None:
        raise ValueError(
            "service_selector is a collection of ids but no id column was found."
        )
    # `{str(v) for v in sel}` is a SET comprehension (set literal `{}` form):
    # the wanted ids as strings, for O(1) membership tests below.
    wanted = {str(v) for v in sel}
    # Compare each row's id (as a string) to the wanted set. The list inside
    # np.array(...) is a LIST comprehension producing one bool per vessel.
    ids = df_week[id_col].astype(str).tolist()
    return np.array([vid in wanted for vid in ids], dtype=bool)


# The bare `*` in the signature means EVERY parameter after it is
# "keyword-only": callers MUST write `vessel_ids=...` etc., not pass them
# positionally. This guards against silently swapping the many same-typed args.
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
    # `np.asarray(x, dtype=...)` returns x as an array of the given dtype,
    # without copying when it already matches. We normalise dtypes here so both
    # the real and synthetic paths feed the solver identical, predictable types.
    arrivals_h = np.asarray(arrivals_h, dtype=np.float32)
    tau_h = np.asarray(tau_h, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    is_service = np.asarray(is_service, dtype=bool)
    latest_start_h = np.asarray(latest_start_h, dtype=np.float32)
    n = len(arrivals_h)

    # Build the (N, B) vessel x berth boolean compatibility matrix; raise if any
    # vessel type matches no berth (allow_unmatched=False = fail loudly).
    compat = vessel_berth_compat(vessel_types, berths, allow_unmatched=False)

    if big_m is None:
        # Auto-size the precedence big-M. `np.max(..., initial=0.0)` returns 0.0
        # for an empty array instead of erroring. Upper bound on any start time:
        # the latest arrival plus all service times stacked at one berth, +1.
        horizon = float(np.max(arrivals_h, initial=0.0))
        big_m = float(horizon + float(tau_h.sum()) + 1.0)

    # Only pass window arrays when at least one service vessel exists.
    # `is_service.any()` is True if any element is True (a vectorised OR).
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
        # List comprehension: stringify every type so the stored metadata is
        # uniformly text regardless of what the source provided.
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

    # `source` is either a CSV path (a str) or an already-loaded DataFrame.
    # `isinstance(x, str)` checks the type. We `.copy()` a passed-in DataFrame so
    # later operations never mutate the caller's object.
    if isinstance(source, str):
        df = pd.read_csv(source)
        provenance = f"csv:{source}"
    else:
        df = source.copy()
        provenance = "dataframe"

    # `a or b` returns `a` if it's truthy else `b`. So each line keeps an
    # explicitly-passed column name, or falls back to auto-detection via _pick.
    # tau_col is only REQUIRED when no `tau` override was supplied.
    arrival_col = arrival_col or _pick(df, _ARRIVAL_ALIASES, "arrival")
    type_col = type_col or _pick(df, _TYPE_ALIASES, "vessel_type")
    berth_col = berth_col or _pick(df, _BERTH_ALIASES, "berth")
    id_col = id_col or _pick(df, _ID_ALIASES, "vessel_id", required=False)
    name_col = name_col or _pick(df, _NAME_ALIASES, "vessel_name", required=False)
    tau_col = tau_col or _pick(df, _TAU_ALIASES, "service_time", required=(tau is None))

    # Derive the berth catalog from the full history (before slicing) so rare
    # vessel types that only appear outside this week still get correct berths.
    if berths is None:
        berths = derive_berths_from_history(
            df, berth_col=berth_col, type_col=type_col, min_count=min_compat_count
        )

    # `pd.to_datetime` parses a column of date strings into real timestamps so
    # we can do arithmetic/comparisons. ws=week start, we=week end (exclusive).
    arr = pd.to_datetime(df[arrival_col])
    ws = pd.to_datetime(week_start)
    we = ws + pd.Timedelta(days=week_days)  # add a duration to a timestamp
    # A boolean MASK: element-wise (arr >= ws) AND (arr < we). `&` is the
    # vectorised AND; each comparison yields a True/False per row. Half-open
    # [ws, we) so back-to-back weeks never double-count a midnight arrival.
    mask = (arr >= ws) & (arr < we)
    df_week = df.loc[mask].copy()  # keep only rows where the mask is True
    if len(df_week) == 0:
        raise ValueError(
            f"No vessel calls with {arrival_col} in [{ws.date()}, {we.date()}). "
            f"Pick a week within the data's coverage."
        )
    # Sort by arrival so vessel index order is chronological; reset_index gives
    # a clean 0..N-1 RangeIndex (drop=True discards the old index column).
    df_week = df_week.sort_values(arrival_col).reset_index(drop=True)
    arr_week = pd.to_datetime(df_week[arrival_col])

    # Convert each arrival to HOURS-since-week-start. Subtracting two datetime
    # series gives a timedelta series; `.dt.total_seconds()` pulls out seconds,
    # /3600 -> hours. `.to_numpy(...)` drops pandas wrapping for a raw array.
    arrivals_h = ((arr_week - ws).dt.total_seconds() / 3600.0).to_numpy(dtype=np.float32)
    arrivals_h = np.maximum(arrivals_h, 0.0)  # guard tiny negatives from rounding

    vessel_types = df_week[type_col].astype(str).tolist()
    # Inline conditional `A if cond else B`: use the id column if we found one,
    # otherwise fall back to integer positions 0..N-1.
    vessel_ids = (
        df_week[id_col].tolist() if id_col else list(range(len(df_week)))
    )
    vessel_names = (
        df_week[name_col].astype(str).tolist() if name_col else [str(v) for v in vessel_ids]
    )

    # Service times τ. Three sources, in priority order:
    #   - None     -> use the true column (the data's actual berth-occupation).
    #   - callable -> e.g. a model's predictions computed from df_week (DFL path).
    #   - array    -> caller-supplied values used as-is.
    if tau is None:
        tau_h = df_week[tau_col].to_numpy(dtype=np.float32)
    elif callable(tau):
        tau_h = np.asarray(tau(df_week), dtype=np.float32)
    else:
        tau_h = np.asarray(tau, dtype=np.float32)
    # Shape guard: tau MUST be exactly (N,) so it index-aligns with the vessels.
    if tau_h.shape != (len(df_week),):
        raise ValueError(f"tau has shape {tau_h.shape}, expected {(len(df_week),)}")

    # Turn the service selector (mask/ids/callable/None) into a length-N bool.
    is_service = _resolve_service_mask(service_selector, df_week, id_col)

    # Weights. Default: base_weight everywhere, then OVERWRITE service rows with
    # service_weight using boolean-mask indexing (`w[is_service] = ...` assigns
    # only the True positions). Or take a caller array/callable instead.
    if weights is None:
        w = np.full(len(df_week), float(base_weight), dtype=np.float32)  # fill array of constant
        w[is_service] = float(service_weight)
    elif callable(weights):
        w = np.asarray(weights(df_week), dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32)

    # Latest-start window lᵢ. Start everyone at +inf (= "no window"), then for
    # service vessels set lᵢ = arrival + slack. Same mask-indexing trick; reading
    # arrivals_h[is_service] selects just the service rows in order.
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
        # `.date()` drops the time-of-day, then str() gives a tidy "YYYY-MM-DD"
        # label. week_end here re-derives the same value as `we` above.
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
    # `np.random.default_rng(seed)` builds a modern numpy random GENERATOR
    # seeded deterministically: same seed -> same week, so demos/tests reproduce.
    rng = np.random.default_rng(seed)
    berths = list(berths)

    # Sample vessel types from the port's rough mix. `.keys()` are the type
    # names; we build a matching probability vector and renormalise it to sum to
    # exactly 1 (rng.choice requires probabilities that sum to 1).
    types_pool = list(_TYPE_MIX.keys())
    probs = np.array([_TYPE_MIX[t] for t in types_pool], dtype=float)
    probs = probs / probs.sum()
    vessel_types = list(rng.choice(types_pool, size=n_vessels, p=probs))

    horizon = week_days * 24.0
    # Arrivals uniformly within the first 90% of the week (leaving room to
    # finish service before the horizon), then SORTED so index order is
    # chronological — matching the real builder's convention.
    arrivals_h = np.sort(rng.uniform(0.0, 0.9 * horizon, size=n_vessels)).astype(np.float32)

    # Service time per vessel = its type's median * lognormal(0, 0.3) noise
    # (always positive, right-skewed). This is a LIST comprehension over types.
    tau_h = np.array(
        [_TYPE_TAU_MEDIAN[t] * float(rng.lognormal(0.0, 0.3)) for t in vessel_types],
        dtype=np.float32,
    )

    # Services: prefer Container vessels (≥2 compatible berths), most-separated.
    # `enumerate` yields (index, value) pairs so we collect the positions of
    # Container vessels.
    container_idx = [i for i, t in enumerate(vessel_types) if t == "Container"]
    # FIX: clamp n_services to a valid range. Without min(...), the non-Container
    # fallback `list(range(n_services))` could index past n_vessels (IndexError)
    # when n_services > n_vessels. Negative requests collapse to 0.
    k = max(0, min(n_services, n_vessels))
    chosen = container_idx[:k] if container_idx else list(range(k))
    is_service = np.zeros(n_vessels, dtype=bool)
    is_service[chosen] = True  # fancy-index assignment: set the chosen rows True

    weights = np.full(n_vessels, float(base_weight), dtype=np.float32)
    weights[is_service] = float(service_weight)

    latest_start_h = np.full(n_vessels, np.inf, dtype=np.float32)
    latest_start_h[is_service] = arrivals_h[is_service] + float(service_slack_hours)

    # f-strings with `:02d` format the integer zero-padded to 2 digits (V00, V01).
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

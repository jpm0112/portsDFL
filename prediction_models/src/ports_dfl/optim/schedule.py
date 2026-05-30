"""Schedule assembly and KPIs for a solved weekly BAP (numpy-only).

Given a solved instance's start times and assignment matrix, build a per-vessel
schedule and compute operational KPIs. Kept free of the solver stack so it can
be unit-tested without PyEPO/Gurobi. The planner script (``scripts/plan_week.py``)
uses these to print tables and draw the Gantt chart.
"""

# `from __future__ import annotations` stores type hints as plain text instead
# of evaluating them at import — lets us write hints like `float | None` freely.
from __future__ import annotations

import numpy as np

# Relative import (leading dot = "from this same package"): we only need the
# bundle's type for hints; this module does not import the solver.
from .weekly_instance import WeeklyInstanceBundle

# A small tolerance for floating-point comparisons. Solver outputs are floats,
# so "equal to zero" is checked as "within _EPS of zero", never exact ==.
_EPS = 1e-3


def berth_index(assignment_row: np.ndarray) -> int:
    """Index of the berth a vessel is assigned to (first entry == 1)."""
    # `assignment_row` is one row of the (N, B) 0/1 matrix. `assignment_row > 0.5`
    # makes a boolean array (True where the value is ~1); `np.flatnonzero` returns
    # the positions of the True entries. We expect exactly one.
    nz = np.flatnonzero(assignment_row > 0.5)
    # If a berth was found return its index (as a plain int); else -1 = unassigned.
    # `nz[0]` is the first matching position; `len(nz)` is how many berths matched.
    return int(nz[0]) if len(nz) else -1


def assemble_schedule(
    bundle: WeeklyInstanceBundle,
    starts: np.ndarray,
    assignment: np.ndarray,
) -> list[dict]:
    """Build a per-vessel schedule table (list of row dicts), sorted by start.

    Each row: index, vessel_id, vessel_name, vessel_type, berth, arrival_h,
    start_h, wait_h, tau_h, finish_h, is_service, latest_start_h, window_ok.
    """
    # Normalise the solver's start times to plain Python floats for safe math.
    starts = np.asarray(starts, dtype=float)
    # We will build one dictionary per vessel and collect them in this list.
    rows: list[dict] = []
    # Loop over every vessel index i (0 .. N-1).
    for i in range(bundle.n_vessels):
        # Which berth did vessel i get? (-1 if somehow unassigned.)
        b = berth_index(assignment[i])
        # Look up the berth's name, or label it UNASSIGNED if none.
        berth_name = bundle.berths[b].name if b >= 0 else "UNASSIGNED"
        start = float(starts[i])
        arrival = float(bundle.arrivals_h[i])
        tau = float(bundle.tau_h[i])
        latest = float(bundle.latest_start_h[i])
        # A service vessel's window is respected if it started no later than its
        # latest allowed start. Non-service vessels have latest = +inf, so this is
        # always True for them. (+_EPS absorbs floating-point noise.)
        window_ok = bool(start <= latest + _EPS)
        # Append this vessel's row. A dict literal `{ "key": value, ... }` maps
        # column names to values; we add it to the list with `.append`.
        rows.append(
            {
                "index": i,
                "vessel_id": bundle.vessel_ids[i],
                "vessel_name": bundle.vessel_names[i],
                "vessel_type": bundle.vessel_types[i],
                "berth": berth_name,
                "arrival_h": arrival,
                "start_h": start,
                # Waiting time = how long after arrival the vessel started; never
                # negative (max with 0 guards against tiny float undershoot).
                "wait_h": max(0.0, start - arrival),
                "tau_h": tau,
                "finish_h": start + tau,
                "is_service": bool(bundle.is_service[i]),
                "latest_start_h": latest,
                "window_ok": window_ok,
            }
        )
    # Sort the rows for readable output: earliest start first, ties broken by
    # berth name. `key=lambda r: (...)` tells sort what to order by; returning a
    # tuple sorts by the first element, then the second on ties.
    rows.sort(key=lambda r: (r["start_h"], r["berth"]))
    return rows


def compute_kpis(
    bundle: WeeklyInstanceBundle,
    starts: np.ndarray,
    assignment: np.ndarray,
    horizon_h: float | None = None,
) -> dict:
    """Operational KPIs for a solved weekly schedule.

    Returns makespan, per-berth utilization, wait statistics, the
    service-vessel no-wait check, and the count of window violations
    (should be 0 in hard-window mode).
    """
    starts = np.asarray(starts, dtype=float)
    # `.astype(float)` returns a float copy of the array (the originals are float32).
    tau = bundle.tau_h.astype(float)
    arrivals = bundle.arrivals_h.astype(float)
    # Vectorised numpy: these operate on whole arrays at once (no Python loop).
    finish = starts + tau                       # completion time of each vessel
    waits = np.maximum(0.0, starts - arrivals)  # per-vessel waiting time, floored at 0
    # Makespan = when the last vessel finishes. Guard the empty case with 0.
    makespan = float(finish.max()) if len(finish) else 0.0
    # Utilization is measured against a horizon: the caller's value if given,
    # otherwise the makespan itself.
    horizon = float(horizon_h) if horizon_h is not None else makespan

    n_berths = len(bundle.berths)
    # `busy[b]` will accumulate the total occupied hours at berth b.
    busy = np.zeros(n_berths)
    for i in range(bundle.n_vessels):
        b = berth_index(assignment[i])
        if b >= 0:
            busy[b] += tau[i]
    # Utilization fraction per berth = busy hours / horizon (avoid /0).
    util = (busy / horizon) if horizon > 0 else np.zeros(n_berths)

    # `svc` is the boolean service mask; `svc.any()` is True if at least one
    # vessel is a service. Boolean-mask indexing `waits[svc]` keeps only the
    # service vessels' waits.
    svc = bundle.is_service
    svc_waits = waits[svc] if svc.any() else np.array([0.0])

    # Window violations: a windowed (service) vessel that started after its lᵢ.
    latest = bundle.latest_start_h.astype(float)
    # `np.isfinite` is False for +inf (non-service) — so `finite` selects only
    # vessels that actually carry a window.
    finite = np.isfinite(latest)
    violations = int(np.sum((starts[finite] > latest[finite] + _EPS))) if finite.any() else 0

    # Return all KPIs in a dict. The `{name: value for ... in ...}` part is a
    # DICT COMPREHENSION: it builds {berth_name: utilization} for every berth.
    return {
        "n_vessels": int(bundle.n_vessels),
        "n_berths": int(n_berths),
        "n_services": int(svc.sum()),
        "makespan_h": makespan,
        "mean_wait_h": float(waits.mean()) if len(waits) else 0.0,
        "max_wait_h": float(waits.max()) if len(waits) else 0.0,
        "service_mean_wait_h": float(svc_waits.mean()),
        "service_max_wait_h": float(svc_waits.max()),
        # True only if no service vessel waited (the no-wait guarantee held).
        "all_services_no_wait": bool(svc_waits.max() <= _EPS) if svc.any() else True,
        "berth_utilization": {bundle.berths[b].name: float(util[b]) for b in range(n_berths)},
        "mean_berth_utilization": float(util.mean()) if n_berths else 0.0,
        "window_violations": violations,
    }

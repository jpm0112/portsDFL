"""Schedule assembly and KPIs for a solved weekly BAP (numpy-only).

Given a solved instance's start times and assignment matrix, build a per-vessel
schedule and compute operational KPIs. Kept free of the solver stack so it can
be unit-tested without PyEPO/Gurobi. The planner script (``scripts/plan_week.py``)
uses these to print tables and draw the Gantt chart.
"""

from __future__ import annotations

import numpy as np

from .weekly_instance import WeeklyInstanceBundle

_EPS = 1e-3


def berth_index(assignment_row: np.ndarray) -> int:
    """Index of the berth a vessel is assigned to (first entry == 1)."""
    nz = np.flatnonzero(assignment_row > 0.5)
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
    starts = np.asarray(starts, dtype=float)
    rows: list[dict] = []
    for i in range(bundle.n_vessels):
        b = berth_index(assignment[i])
        berth_name = bundle.berths[b].name if b >= 0 else "UNASSIGNED"
        start = float(starts[i])
        arrival = float(bundle.arrivals_h[i])
        tau = float(bundle.tau_h[i])
        latest = float(bundle.latest_start_h[i])
        window_ok = bool(start <= latest + _EPS)  # inf for non-service => always ok
        rows.append(
            {
                "index": i,
                "vessel_id": bundle.vessel_ids[i],
                "vessel_name": bundle.vessel_names[i],
                "vessel_type": bundle.vessel_types[i],
                "berth": berth_name,
                "arrival_h": arrival,
                "start_h": start,
                "wait_h": max(0.0, start - arrival),
                "tau_h": tau,
                "finish_h": start + tau,
                "is_service": bool(bundle.is_service[i]),
                "latest_start_h": latest,
                "window_ok": window_ok,
            }
        )
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
    tau = bundle.tau_h.astype(float)
    arrivals = bundle.arrivals_h.astype(float)
    finish = starts + tau
    waits = np.maximum(0.0, starts - arrivals)
    makespan = float(finish.max()) if len(finish) else 0.0
    horizon = float(horizon_h) if horizon_h is not None else makespan

    n_berths = len(bundle.berths)
    busy = np.zeros(n_berths)
    for i in range(bundle.n_vessels):
        b = berth_index(assignment[i])
        if b >= 0:
            busy[b] += tau[i]
    util = (busy / horizon) if horizon > 0 else np.zeros(n_berths)

    svc = bundle.is_service
    svc_waits = waits[svc] if svc.any() else np.array([0.0])

    # Window violations: a windowed (service) vessel started after lᵢ.
    latest = bundle.latest_start_h.astype(float)
    finite = np.isfinite(latest)
    violations = int(np.sum((starts[finite] > latest[finite] + _EPS))) if finite.any() else 0

    return {
        "n_vessels": int(bundle.n_vessels),
        "n_berths": int(n_berths),
        "n_services": int(svc.sum()),
        "makespan_h": makespan,
        "mean_wait_h": float(waits.mean()) if len(waits) else 0.0,
        "max_wait_h": float(waits.max()) if len(waits) else 0.0,
        "service_mean_wait_h": float(svc_waits.mean()),
        "service_max_wait_h": float(svc_waits.max()),
        "all_services_no_wait": bool(svc_waits.max() <= _EPS) if svc.any() else True,
        "berth_utilization": {bundle.berths[b].name: float(util[b]) for b in range(n_berths)},
        "mean_berth_utilization": float(util.mean()) if n_berths else 0.0,
        "window_violations": violations,
    }

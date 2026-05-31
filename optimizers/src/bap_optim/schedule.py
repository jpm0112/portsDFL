"""Schedule assembly and KPIs for a solved weekly BAP (numpy-only).

Given a solved instance's start times and assignment matrix, build a per-vessel
schedule and compute operational KPIs. Kept free of the solver stack so it can
be unit-tested without PyEPO/Gurobi. The planner script (``scripts/plan_week.py``)
uses these to print tables and draw the Gantt chart.
"""

from __future__ import annotations

import numpy as np

from .weekly_instance import WeeklyInstanceBundle

# Tolerance for float comparisons: solver outputs are floats, so "equal to
# zero" is checked as "within _EPS", never exact ==.
_EPS = 1e-3


def berth_index(assignment_row: np.ndarray) -> int:
    """Index of the berth a vessel is assigned to (first entry == 1)."""
    nz = np.flatnonzero(assignment_row > 0.5)
    return int(nz[0]) if len(nz) else -1  # -1 = unassigned


def assemble_schedule(
    bundle: WeeklyInstanceBundle,
    starts: np.ndarray,
    assignment: np.ndarray,
    ein: np.ndarray | None = None,
    eout: np.ndarray | None = None,
) -> list[dict]:
    """Build a per-vessel schedule table (list of row dicts), sorted by start.

    Each row: index, vessel_id, vessel_name, vessel_type, berth, arrival_h,
    start_h, wait_h, tau_h, finish_h, is_service, latest_start_h, window_ok.

    When the channel transit starts ``ein``/``eout`` are supplied (channel
    modelled), each row also carries ``enter_h`` (inbound transit start),
    ``exit_h`` (outbound transit start) and ``departure_h`` (``exit_h`` + channel
    transit time — when the vessel actually clears the port).
    """
    starts = np.asarray(starts, dtype=float)
    c = bundle.channel_time
    has_channel = c is not None and ein is not None and eout is not None
    if has_channel:
        ein = np.asarray(ein, dtype=float)
        eout = np.asarray(eout, dtype=float)
    rows: list[dict] = []
    for i in range(bundle.n_vessels):
        b = berth_index(assignment[i])
        berth_name = bundle.berths[b].name if b >= 0 else "UNASSIGNED"
        start = float(starts[i])
        arrival = float(bundle.arrivals_h[i])
        tau = float(bundle.tau_h[i])
        latest = float(bundle.latest_start_h[i])
        # The no-wait window sits on ENTRY (ein) when a channel is modelled, else
        # on the berth start s. Non-service vessels have latest = +inf -> always
        # True. +_EPS absorbs solver float noise.
        win_var = float(ein[i]) if has_channel else start
        window_ok = bool(win_var <= latest + _EPS)
        row = {
            "index": i,
            "vessel_id": bundle.vessel_ids[i],
            "vessel_name": bundle.vessel_names[i],
            "vessel_type": bundle.vessel_types[i],
            "berth": berth_name,
            "arrival_h": arrival,
            "start_h": start,
            # Idle wait = time from arrival to berthing, minus the mandatory
            # inbound transit when a channel is modelled (so a vessel that enters
            # on arrival and berths straight after transit shows wait 0, not c).
            # max with 0 guards against tiny float undershoot.
            "wait_h": max(0.0, start - arrival - (float(c) if has_channel else 0.0)),
            "tau_h": tau,
            "finish_h": start + tau,
            "is_service": bool(bundle.is_service[i]),
            "latest_start_h": latest,
            "window_ok": window_ok,
        }
        if has_channel:
            row["enter_h"] = float(ein[i])
            row["exit_h"] = float(eout[i])
            row["departure_h"] = float(eout[i]) + float(c)
        rows.append(row)
    # Earliest start first, ties broken by berth name.
    rows.sort(key=lambda r: (r["start_h"], r["berth"]))
    return rows


def compute_kpis(
    bundle: WeeklyInstanceBundle,
    starts: np.ndarray,
    assignment: np.ndarray,
    horizon_h: float | None = None,
    ein: np.ndarray | None = None,
    eout: np.ndarray | None = None,
) -> dict:
    """Operational KPIs for a solved weekly schedule.

    Returns makespan, per-berth utilization, wait statistics, the
    service-vessel no-wait check, and the count of window violations
    (should be 0 in hard-window mode).

    When the channel transit starts ``ein``/``eout`` are supplied, also returns
    a ``channel`` block: transit time, busy hours, utilization over the horizon,
    actual port makespan (last departure), and a no-overlap check — the number
    of overlapping transit pairs, which must be 0 for a valid schedule.
    """
    starts = np.asarray(starts, dtype=float)
    tau = bundle.tau_h.astype(float)
    arrivals = bundle.arrivals_h.astype(float)
    finish = starts + tau
    c = bundle.channel_time
    channel_on = c is not None and ein is not None and eout is not None
    if channel_on:
        c = float(c)
        ein = np.asarray(ein, dtype=float)
        eout = np.asarray(eout, dtype=float)
    # Idle wait excludes the mandatory inbound transit when a channel is modelled.
    waits = np.maximum(0.0, starts - arrivals - (c if channel_on else 0.0))
    makespan = float(finish.max()) if len(finish) else 0.0
    # Utilization horizon: the caller's value if given, else the makespan.
    horizon = float(horizon_h) if horizon_h is not None else makespan

    n_berths = len(bundle.berths)
    busy = np.zeros(n_berths)
    for i in range(bundle.n_vessels):
        b = berth_index(assignment[i])
        if b >= 0:
            busy[b] += tau[i]
    util = (busy / horizon) if horizon > 0 else np.zeros(n_berths)  # avoid /0

    svc = bundle.is_service
    svc_waits = waits[svc] if svc.any() else np.array([0.0])

    # Window violations: a windowed service vessel that breached its lᵢ. The
    # window sits on ENTRY (ein) when a channel is modelled, else on berth start.
    latest = bundle.latest_start_h.astype(float)
    win_var = ein if channel_on else starts
    # isfinite is False for +inf (non-service), so `finite` selects windowed vessels.
    finite = np.isfinite(latest)
    violations = int(np.sum((win_var[finite] > latest[finite] + _EPS))) if finite.any() else 0

    kpis = {
        "n_vessels": int(bundle.n_vessels),
        "n_berths": int(n_berths),
        "n_services": int(svc.sum()),
        "makespan_h": makespan,
        "mean_wait_h": float(waits.mean()) if len(waits) else 0.0,
        "max_wait_h": float(waits.max()) if len(waits) else 0.0,
        "service_mean_wait_h": float(svc_waits.mean()),
        "service_max_wait_h": float(svc_waits.max()),
        # True only if no service vessel waited (no-wait guarantee held).
        "all_services_no_wait": bool(svc_waits.max() <= _EPS) if svc.any() else True,
        "berth_utilization": {bundle.berths[b].name: float(util[b]) for b in range(n_berths)},
        "mean_berth_utilization": float(util.mean()) if n_berths else 0.0,
        "window_violations": violations,
    }

    # Channel KPIs: each vessel transits twice (in + out), each lasting c hours.
    # c/ein/eout were normalised at the top when channel_on.
    if channel_on:
        # Every transit occupies the channel for c hours; 2N transits total.
        n_transits = 2 * bundle.n_vessels
        busy_ch = n_transits * c
        # Port makespan = last vessel actually clears the channel on its way out.
        port_makespan = float((eout + c).max()) if bundle.n_vessels else 0.0
        ch_horizon = float(horizon_h) if horizon_h is not None else port_makespan
        # Count overlapping transit pairs: sort all 2N intervals [t, t+c] by start
        # and flag any neighbour that begins before its predecessor ends.
        intervals = sorted([float(t) for t in ein] + [float(t) for t in eout])
        overlaps = sum(1 for a, b in zip(intervals, intervals[1:]) if b < a + c - _EPS)
        kpis["channel"] = {
            "transit_time_h": c,
            "n_transits": int(n_transits),
            "busy_h": float(busy_ch),
            "utilization": float(busy_ch / ch_horizon) if ch_horizon > 0 else 0.0,
            "port_makespan_h": port_makespan,
            "overlaps": int(overlaps),
            "no_overlap": bool(overlaps == 0),
        }
    return kpis

"""Tests for the (solver-free) weekly-instance builder, berth compatibility,
and schedule/KPI helpers.

These exercise the pre-solve pipeline and the post-solve reporting math without
constructing a MILP, so they run in any environment with numpy + pandas (no
PyEPO/Gurobi needed).
"""

import numpy as np
import pytest

from ports_dfl.optim.berths import (
    DEFAULT_BERTHS,
    Berth,
    berth_names,
    derive_berths_from_history,
    vessel_berth_compat,
)
from ports_dfl.optim.instance import BAPInstance
from ports_dfl.optim.schedule import assemble_schedule, compute_kpis
from ports_dfl.optim.weekly_instance import (
    build_weekly_instance,
    generate_synthetic_weekly_instance,
)


# --- berths / compatibility -------------------------------------------------

def test_default_compat_routes_types_to_expected_berths() -> None:
    """Liquid bulk -> QC only; container -> STI & DP World; dry bulk -> PANUL."""
    types = ["Container", "Liquid Bulk", "Dry Bulk"]
    compat = vessel_berth_compat(types, DEFAULT_BERTHS)
    names = berth_names(DEFAULT_BERTHS)
    qc = names.index("QC")
    panul = names.index("PANUL")
    sti = names.index("STI")
    dpw = names.index("DP World")
    # container
    assert compat[0, sti] and compat[0, dpw] and not compat[0, qc] and not compat[0, panul]
    # liquid bulk -> QC only
    assert compat[1, qc] and compat[1].sum() == 1
    # dry bulk -> PANUL only
    assert compat[2, panul] and compat[2].sum() == 1


def test_compat_unmatched_type_raises_unless_allowed() -> None:
    """A type served by no berth raises by default; allow_unmatched routes to all."""
    with pytest.raises(ValueError):
        vessel_berth_compat(["Submarine"], DEFAULT_BERTHS)
    compat = vessel_berth_compat(["Submarine"], DEFAULT_BERTHS, allow_unmatched=True)
    assert compat.all()


def test_derive_berths_from_history() -> None:
    """Served-type sets are read from (berth x type) co-occurrence."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "Terminal": ["STI", "STI", "QC", "PANUL", "QC"],
            "vessel_type_group": ["Container", "Container", "Liquid Bulk", "Dry Bulk", "Liquid Bulk"],
        }
    )
    berths = derive_berths_from_history(df, berth_col="Terminal", type_col="vessel_type_group")
    by_name = {b.name: b for b in berths}
    assert by_name["STI"].served_types == frozenset({"Container"})
    assert by_name["QC"].served_types == frozenset({"Liquid Bulk"})
    assert by_name["PANUL"].served_types == frozenset({"Dry Bulk"})
    # min_count drops rare one-offs
    berths2 = derive_berths_from_history(df, min_count=2)
    assert {b.name for b in berths2 if b.served_types} == {"STI", "QC"}


# --- synthetic builder ------------------------------------------------------

def test_synthetic_instance_shapes_and_windows() -> None:
    b = generate_synthetic_weekly_instance(n_vessels=15, n_services=3, seed=7)
    inst = b.instance
    assert inst.n_vessels == 15
    assert inst.berth_compat.shape == (15, inst.n_berths)
    # every vessel has at least one compatible berth
    assert bool(inst.berth_compat.any(axis=1).all())
    assert int(b.is_service.sum()) == 3
    # services carry a finite window; others are inf
    assert np.isfinite(b.latest_start_h[b.is_service]).all()
    assert np.isinf(b.latest_start_h[~b.is_service]).all()
    # service weight elevated
    assert (b.weights[b.is_service] > b.weights[~b.is_service].max() - 1e-6).all()


# --- real-data builder (synthetic dataframe, build-script schema) ----------

def _toy_calls_df():
    import pandas as pd

    return pd.DataFrame(
        {
            "Cód. nave": [1, 2, 3, 4, 5, 6],
            "Nave": ["A", "B", "C", "D", "E", "F"],
            "Terminal": ["STI", "STI", "QC", "STI", "QC", "STI"],
            "vessel_type_group": [
                "Container", "Container", "Liquid Bulk", "Container", "Liquid Bulk", "Container",
            ],
            "estadia_sitio_hours": [30.0, 24.0, 20.0, 28.0, 18.0, 22.0],
            "F. arribo": [
                "2025-03-03 02:00", "2025-03-04 05:00", "2025-03-05 10:00",
                "2025-03-09 12:00", "2025-03-12 00:00", "2025-02-28 00:00",
            ],
        }
    )


def test_build_weekly_instance_slices_one_week() -> None:
    df = _toy_calls_df()
    bundle = build_weekly_instance(
        df, "2025-03-03", week_days=7, service_selector=[2], service_slack_hours=0.0
    )
    inst = bundle.instance
    # ids 1..4 fall in [03-03, 03-10); id 5 (03-12) and id 6 (02-28) excluded
    assert inst.n_vessels == 4
    assert set(map(int, bundle.vessel_ids)) == {1, 2, 3, 4}
    # arrivals are within the 168h window and sorted
    assert (bundle.arrivals_h >= 0).all() and (bundle.arrivals_h < 168.0).all()
    assert (np.diff(bundle.arrivals_h) >= 0).all()
    # tau comes from estadia_sitio_hours
    idx_by_id = {int(v): k for k, v in enumerate(bundle.vessel_ids)}
    assert bundle.tau_h[idx_by_id[1]] == pytest.approx(30.0)
    # service = vessel id 2, with a finite window and elevated weight
    svc_k = idx_by_id[2]
    assert bundle.is_service[svc_k] and np.isfinite(bundle.latest_start_h[svc_k])
    assert bundle.is_service.sum() == 1
    assert bundle.weights[svc_k] > bundle.weights[idx_by_id[1]]
    # berths derived from history; liquid-bulk vessel only compatible with QC
    names = [b.name for b in bundle.berths]
    qc = names.index("QC")
    liquid_k = idx_by_id[3]
    assert inst.berth_compat[liquid_k, qc] and inst.berth_compat[liquid_k].sum() == 1


def test_build_weekly_instance_empty_week_raises() -> None:
    df = _toy_calls_df()
    with pytest.raises(ValueError):
        build_weekly_instance(df, "2024-01-01", week_days=7)


# --- schedule + KPIs --------------------------------------------------------

def test_schedule_and_kpis_on_handmade_solution() -> None:
    """Given starts + assignment, schedule rows and KPIs are computed correctly."""
    # 3 vessels, 2 berths. Vessel 0 service (no-wait window at t=0).
    berths = [Berth("B0", frozenset({"X"})), Berth("B1", frozenset({"X"}))]
    compat = np.ones((3, 2), dtype=bool)
    inst = BAPInstance(
        n_vessels=3, n_berths=2,
        arrivals=np.array([0.0, 1.0, 2.0], np.float32),
        weights=np.array([3.0, 1.0, 1.0], np.float32),
        big_m=100.0,
        latest_start=np.array([0.0, np.inf, np.inf], np.float32),
        berth_compat=compat,
        service=np.array([True, False, False]),
    )
    from ports_dfl.optim.weekly_instance import WeeklyInstanceBundle

    bundle = WeeklyInstanceBundle(
        instance=inst, berths=berths,
        vessel_ids=[0, 1, 2], vessel_names=["a", "b", "c"],
        vessel_types=["X", "X", "X"],
        arrivals_h=inst.arrivals, tau_h=np.array([10.0, 5.0, 6.0], np.float32),
        weights=inst.weights, is_service=inst.service,
        latest_start_h=inst.latest_start, week_start="w", week_end="w", source="test",
    )
    # Solution: v0->B0 at t0 (service, no wait), v1->B1 at t1, v2->B0 after v0 at t10
    starts = np.array([0.0, 1.0, 10.0], np.float32)
    assignment = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)

    rows = assemble_schedule(bundle, starts, assignment)
    assert len(rows) == 3
    kpis = compute_kpis(bundle, starts, assignment, horizon_h=24.0)
    assert kpis["all_services_no_wait"] is True
    assert kpis["service_max_wait_h"] == pytest.approx(0.0)
    assert kpis["window_violations"] == 0
    assert kpis["makespan_h"] == pytest.approx(16.0)  # v2 finishes at 10+6
    # B0 busy 16h, B1 busy 5h over 24h horizon
    assert kpis["berth_utilization"]["B0"] == pytest.approx(16.0 / 24.0)
    assert kpis["berth_utilization"]["B1"] == pytest.approx(5.0 / 24.0)

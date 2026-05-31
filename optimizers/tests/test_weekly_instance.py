"""Tests for the (solver-free) weekly-instance builder, berth compatibility,
and schedule/KPI helpers.

These exercise the pre-solve pipeline and the post-solve reporting math without
constructing a MILP, so they run in any environment with numpy + pandas (no
PyEPO/Gurobi needed).
"""

import numpy as np
import pytest

from bap_optim.berths import (
    DEFAULT_BERTHS,
    Berth,
    berth_names,
    derive_berths_from_history,
    vessel_berth_compat,
)
from bap_optim.instance import BAPInstance
from bap_optim.schedule import assemble_schedule, compute_kpis
from bap_optim.weekly_instance import (
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
    # Container (row 0): allowed at STI and DP World, NOT at QC or PANUL.
    assert compat[0, sti] and compat[0, dpw] and not compat[0, qc] and not compat[0, panul]
    # Liquid bulk -> QC only (compatible with exactly one berth).
    assert compat[1, qc] and compat[1].sum() == 1
    # Dry bulk -> PANUL only.
    assert compat[2, panul] and compat[2].sum() == 1


def test_compat_unmatched_type_raises_unless_allowed() -> None:
    """A type served by no berth raises by default; allow_unmatched routes to all."""
    # "Submarine" matches no berth, so the default behaviour must raise.
    with pytest.raises(ValueError):
        vessel_berth_compat(["Submarine"], DEFAULT_BERTHS)
    # With the permissive flag, an unmatched type is routed to every berth.
    compat = vessel_berth_compat(["Submarine"], DEFAULT_BERTHS, allow_unmatched=True)
    assert compat.all()


def test_derive_berths_from_history() -> None:
    """Served-type sets are read from (berth x type) co-occurrence."""
    import pandas as pd

    # Fake call log: STI saw Container twice, QC saw Liquid Bulk twice, PANUL saw
    # Dry Bulk once.
    df = pd.DataFrame(
        {
            "Terminal": ["STI", "STI", "QC", "PANUL", "QC"],
            "vessel_type_group": ["Container", "Container", "Liquid Bulk", "Dry Bulk", "Liquid Bulk"],
        }
    )
    berths = derive_berths_from_history(df, berth_col="Terminal", type_col="vessel_type_group")
    by_name = {b.name: b for b in berths}
    # Each berth's served-type set matches what the history showed.
    assert by_name["STI"].served_types == frozenset({"Container"})
    assert by_name["QC"].served_types == frozenset({"Liquid Bulk"})
    assert by_name["PANUL"].served_types == frozenset({"Dry Bulk"})
    # min_count=2 keeps only types served at least twice; PANUL's single Dry Bulk
    # call falls below the threshold, leaving it with an empty served-type set.
    berths2 = derive_berths_from_history(df, min_count=2)
    assert {b.name for b in berths2 if b.served_types} == {"STI", "QC"}


# --- synthetic builder ------------------------------------------------------

def test_synthetic_instance_shapes_and_windows() -> None:
    b = generate_synthetic_weekly_instance(n_vessels=15, n_services=3, seed=7)
    inst = b.instance
    assert inst.n_vessels == 15
    assert inst.berth_compat.shape == (15, inst.n_berths)
    # Every vessel has at least one compatible berth (else infeasible).
    assert bool(inst.berth_compat.any(axis=1).all())
    assert int(b.is_service.sum()) == 3
    # Services carry a finite window; non-services have an infinite (no-window) one.
    assert np.isfinite(b.latest_start_h[b.is_service]).all()
    assert np.isinf(b.latest_start_h[~b.is_service]).all()
    # Every service weight must exceed the largest non-service weight (-1e-6 so an
    # exact tie doesn't fail on float rounding).
    assert (b.weights[b.is_service] > b.weights[~b.is_service].max() - 1e-6).all()


# --- real-data builder (synthetic dataframe, build-script schema) ----------

# Fake call log reused by the two tests below.
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
    # service_selector=[2] marks vessel id 2 as a priority service; slack 0 = no wait.
    bundle = build_weekly_instance(
        df, "2025-03-03", week_days=7, service_selector=[2], service_slack_hours=0.0
    )
    inst = bundle.instance
    # Window is half-open [03-03, 03-10): ids 1..4 inside; id 5 (03-12) after and
    # id 6 (02-28) before are excluded.
    assert inst.n_vessels == 4
    assert set(map(int, bundle.vessel_ids)) == {1, 2, 3, 4}
    # Arrivals within the 168h window and sorted ascending (builder sorts by time).
    assert (bundle.arrivals_h >= 0).all() and (bundle.arrivals_h < 168.0).all()
    assert (np.diff(bundle.arrivals_h) >= 0).all()
    # Slice is re-sorted by arrival, so row index != id; map ids -> row index.
    idx_by_id = {int(v): k for k, v in enumerate(bundle.vessel_ids)}
    # tau comes from the estadia_sitio_hours column (approx absorbs float32 rounding).
    assert bundle.tau_h[idx_by_id[1]] == pytest.approx(30.0)
    # service = vessel id 2, with a finite window and elevated weight.
    svc_k = idx_by_id[2]
    assert bundle.is_service[svc_k] and np.isfinite(bundle.latest_start_h[svc_k])
    assert bundle.is_service.sum() == 1
    assert bundle.weights[svc_k] > bundle.weights[idx_by_id[1]]
    # Berths derived from the full history; the liquid-bulk vessel (id 3) must be
    # compatible with QC and QC only.
    names = [b.name for b in bundle.berths]
    qc = names.index("QC")
    liquid_k = idx_by_id[3]
    assert inst.berth_compat[liquid_k, qc] and inst.berth_compat[liquid_k].sum() == 1


def test_build_weekly_instance_empty_week_raises() -> None:
    df = _toy_calls_df()
    # Toy data is all 2025, so a 2024 week has zero calls; the builder must raise
    # rather than silently return an empty instance.
    with pytest.raises(ValueError):
        build_weekly_instance(df, "2024-01-01", week_days=7)


# --- schedule + KPIs --------------------------------------------------------

def test_schedule_and_kpis_on_handmade_solution() -> None:
    """Given starts + assignment, schedule rows and KPIs are computed correctly."""
    # 3 vessels, 2 berths. Vessel 0 is a service with a no-wait window at t=0.
    # Hand-built KNOWN solution so the expected KPIs are easy to verify by hand.
    berths = [Berth("B0", frozenset({"X"})), Berth("B1", frozenset({"X"}))]
    compat = np.ones((3, 2), dtype=bool)  # every vessel may use either berth
    inst = BAPInstance(
        n_vessels=3, n_berths=2,
        arrivals=np.array([0.0, 1.0, 2.0], np.float32),
        weights=np.array([3.0, 1.0, 1.0], np.float32),
        big_m=100.0,
        # Vessel 0 must start by t=0 (no-wait); the others have inf = no window.
        latest_start=np.array([0.0, np.inf, np.inf], np.float32),
        berth_compat=compat,
        service=np.array([True, False, False]),
    )
    from bap_optim.weekly_instance import WeeklyInstanceBundle

    bundle = WeeklyInstanceBundle(
        instance=inst, berths=berths,
        vessel_ids=[0, 1, 2], vessel_names=["a", "b", "c"],
        vessel_types=["X", "X", "X"],
        arrivals_h=inst.arrivals, tau_h=np.array([10.0, 5.0, 6.0], np.float32),
        weights=inst.weights, is_service=inst.service,
        latest_start_h=inst.latest_start, week_start="w", week_end="w", source="test",
    )
    # Hand-made SOLUTION we are scoring:
    #   v0 -> B0 at t=0 (service, starts on arrival), v1 -> B1 at t=1,
    #   v2 -> B0 at t=10 (after v0 frees B0).
    starts = np.array([0.0, 1.0, 10.0], np.float32)
    # One-hot rows: v0->B0 (col 0), v1->B1 (col 1), v2->B0 (col 0).
    assignment = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)

    rows = assemble_schedule(bundle, starts, assignment)
    assert len(rows) == 3
    kpis = compute_kpis(bundle, starts, assignment, horizon_h=24.0)
    # The lone service (v0) started exactly on arrival, so it never waited.
    assert kpis["all_services_no_wait"] is True
    assert kpis["service_max_wait_h"] == pytest.approx(0.0)
    assert kpis["window_violations"] == 0
    # makespan = v2's finish = start 10 + tau 6 = 16.
    assert kpis["makespan_h"] == pytest.approx(16.0)
    # B0 hosted v0 (tau 10) + v2 (tau 6) = 16h busy; B1 hosted v1 (tau 5) = 5h.
    assert kpis["berth_utilization"]["B0"] == pytest.approx(16.0 / 24.0)
    assert kpis["berth_utilization"]["B1"] == pytest.approx(5.0 / 24.0)

"""Tests for the (solver-free) weekly-instance builder, berth compatibility,
and schedule/KPI helpers.

These exercise the pre-solve pipeline and the post-solve reporting math without
constructing a MILP, so they run in any environment with numpy + pandas (no
PyEPO/Gurobi needed).
"""

# `import x` / `from x import y` pull in code from other modules so we can use it.
# numpy ("np") is for numeric arrays; pytest is the test framework that runs this file.
import numpy as np
import pytest

# These imports bring in the functions/classes we are TESTING from the ports_dfl
# package. If any of these names didn't exist, the import would fail and pytest
# would report a "collection error" for the whole file.
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

# pytest auto-discovers any function whose name starts with `test_` and runs it
# as a test. `-> None` is a type hint saying "this function returns nothing"; it
# does not affect behaviour. The test PASSES if it finishes without an `assert`
# failing or an exception being raised.
def test_default_compat_routes_types_to_expected_berths() -> None:
    """Liquid bulk -> QC only; container -> STI & DP World; dry bulk -> PANUL."""
    # ARRANGE: set up the inputs. Three vessels, one of each type.
    types = ["Container", "Liquid Bulk", "Dry Bulk"]
    # ACT: call the function under test to get the (vessel x berth) bool matrix.
    compat = vessel_berth_compat(types, DEFAULT_BERTHS)
    # Look up each berth's COLUMN index by name, so the asserts below read clearly.
    # `.index("QC")` returns the position of "QC" in the names list.
    names = berth_names(DEFAULT_BERTHS)
    qc = names.index("QC")
    panul = names.index("PANUL")
    sti = names.index("STI")
    dpw = names.index("DP World")
    # ASSERT: an `assert <expr>` makes the test FAIL if <expr> is False/falsy.
    # Container (row 0): allowed at STI and DP World, NOT at QC or PANUL.
    assert compat[0, sti] and compat[0, dpw] and not compat[0, qc] and not compat[0, panul]
    # liquid bulk -> QC only. `.sum()` on a bool row counts the True cells; == 1
    # means it is compatible with EXACTLY one berth (and we checked that it's QC).
    assert compat[1, qc] and compat[1].sum() == 1
    # dry bulk -> PANUL only (again exactly one compatible berth).
    assert compat[2, panul] and compat[2].sum() == 1


def test_compat_unmatched_type_raises_unless_allowed() -> None:
    """A type served by no berth raises by default; allow_unmatched routes to all."""
    # `with pytest.raises(ValueError):` asserts that the code INSIDE the block
    # raises a ValueError. If no error (or a different error) is raised, the test
    # FAILS. "Submarine" matches no berth, so the default behaviour must raise.
    with pytest.raises(ValueError):
        vessel_berth_compat(["Submarine"], DEFAULT_BERTHS)
    # With the permissive flag, an unmatched type is routed to EVERY berth instead.
    compat = vessel_berth_compat(["Submarine"], DEFAULT_BERTHS, allow_unmatched=True)
    # `.all()` is True only if every cell of the matrix is True (all berths allowed).
    assert compat.all()


def test_derive_berths_from_history() -> None:
    """Served-type sets are read from (berth x type) co-occurrence."""
    # Local import (inside the function): pandas is only needed for this test, so
    # importing it here keeps the other tests runnable even if pandas were absent.
    import pandas as pd

    # ARRANGE: a tiny fake call log. Each row is one historical vessel call:
    # (which terminal it used, what type it was). STI saw Container twice, QC saw
    # Liquid Bulk twice, PANUL saw Dry Bulk once.
    df = pd.DataFrame(
        {
            "Terminal": ["STI", "STI", "QC", "PANUL", "QC"],
            "vessel_type_group": ["Container", "Container", "Liquid Bulk", "Dry Bulk", "Liquid Bulk"],
        }
    )
    # ACT: derive the berth catalog (one Berth per terminal) from that history.
    berths = derive_berths_from_history(df, berth_col="Terminal", type_col="vessel_type_group")
    # A DICT COMPREHENSION `{key: value for x in seq}`: index the berths by name
    # so we can look one up as `by_name["STI"]`.
    by_name = {b.name: b for b in berths}
    # ASSERT: each berth's served-type set matches what the history showed.
    # `frozenset({...})` is an immutable set; equality ignores order.
    assert by_name["STI"].served_types == frozenset({"Container"})
    assert by_name["QC"].served_types == frozenset({"Liquid Bulk"})
    assert by_name["PANUL"].served_types == frozenset({"Dry Bulk"})
    # min_count=2 keeps only types served at least twice, dropping rare one-offs.
    # PANUL's single Dry Bulk call falls below the threshold, so PANUL ends up
    # with an EMPTY served-type set and is filtered out below.
    berths2 = derive_berths_from_history(df, min_count=2)
    # SET COMPREHENSION with a filter: collect names of berths that still have a
    # non-empty served_types set. Only STI and QC had a type seen >= 2 times.
    assert {b.name for b in berths2 if b.served_types} == {"STI", "QC"}


# --- synthetic builder ------------------------------------------------------

def test_synthetic_instance_shapes_and_windows() -> None:
    # seed=7 makes the random generator deterministic, so this test is reproducible.
    b = generate_synthetic_weekly_instance(n_vessels=15, n_services=3, seed=7)
    inst = b.instance
    # The instance should report exactly the number of vessels we asked for.
    assert inst.n_vessels == 15
    # `.shape` is the (rows, cols) size of the compat matrix: one row per vessel,
    # one column per berth.
    assert inst.berth_compat.shape == (15, inst.n_berths)
    # every vessel has at least one compatible berth.
    # `.any(axis=1)` collapses each ROW to "is any berth allowed?"; `.all()` then
    # requires that to hold for every vessel (else the instance is infeasible).
    assert bool(inst.berth_compat.any(axis=1).all())
    # `.sum()` on the bool service mask counts the service vessels; we asked for 3.
    assert int(b.is_service.sum()) == 3
    # services carry a finite window; others are inf.
    # `b.latest_start_h[b.is_service]` is boolean-mask indexing: keep only the
    # service rows. `np.isfinite(...).all()` requires every one to be a real number.
    assert np.isfinite(b.latest_start_h[b.is_service]).all()
    # `~b.is_service` flips the mask (True<->False) to select NON-service vessels;
    # all of those must have an infinite (= "no window") latest start.
    assert np.isinf(b.latest_start_h[~b.is_service]).all()
    # service weight elevated: every service vessel's weight must exceed the
    # largest non-service weight. The `- 1e-6` is a tiny float tolerance so an
    # exact tie doesn't fail due to floating-point rounding.
    assert (b.weights[b.is_service] > b.weights[~b.is_service].max() - 1e-6).all()


# --- real-data builder (synthetic dataframe, build-script schema) ----------

# A plain helper (NOT a test — its name does not start with `test_`, so pytest
# won't run it directly). It returns a small fake call log reused by the two
# tests below. Building it in one place avoids copy-pasting the data.
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
    # ACT: slice the 7-day window starting 2025-03-03. `service_selector=[2]` marks
    # the vessel with id 2 as a priority "service"; slack 0 = it must not wait.
    bundle = build_weekly_instance(
        df, "2025-03-03", week_days=7, service_selector=[2], service_slack_hours=0.0
    )
    inst = bundle.instance
    # The window is half-open [03-03, 03-10): ids 1..4 arrive inside it; id 5
    # (03-12) is after and id 6 (02-28) is before, so both are excluded.
    assert inst.n_vessels == 4
    # `map(int, ...)` converts each id to int; `set(...)` ignores order, so this
    # just checks WHICH ids survived the slice (not their order).
    assert set(map(int, bundle.vessel_ids)) == {1, 2, 3, 4}
    # arrivals are within the 168h (=7*24) window and sorted ascending.
    assert (bundle.arrivals_h >= 0).all() and (bundle.arrivals_h < 168.0).all()
    # `np.diff` gives consecutive differences; all >= 0 means non-decreasing
    # (the builder sorts vessels chronologically).
    assert (np.diff(bundle.arrivals_h) >= 0).all()
    # Because the slice is re-sorted by arrival, a vessel's ROW index is not its
    # id. Build a {vessel_id -> row index} map so the asserts can refer to ids.
    idx_by_id = {int(v): k for k, v in enumerate(bundle.vessel_ids)}
    # tau (service time) comes straight from the estadia_sitio_hours column.
    # `pytest.approx(30.0)` compares floats with a small tolerance, so tiny
    # rounding from the float32 conversion won't cause a spurious failure.
    assert bundle.tau_h[idx_by_id[1]] == pytest.approx(30.0)
    # service = vessel id 2, with a finite window and elevated weight.
    svc_k = idx_by_id[2]
    assert bundle.is_service[svc_k] and np.isfinite(bundle.latest_start_h[svc_k])
    # Exactly one vessel was selected as a service.
    assert bundle.is_service.sum() == 1
    # The service vessel's weight is boosted above an ordinary vessel's (id 1).
    assert bundle.weights[svc_k] > bundle.weights[idx_by_id[1]]
    # berths derived from the FULL history; the liquid-bulk vessel (id 3) must be
    # compatible with QC and with QC only (exactly one True in its compat row).
    names = [b.name for b in bundle.berths]
    qc = names.index("QC")
    liquid_k = idx_by_id[3]
    assert inst.berth_compat[liquid_k, qc] and inst.berth_compat[liquid_k].sum() == 1


def test_build_weekly_instance_empty_week_raises() -> None:
    df = _toy_calls_df()
    # The toy data is all in 2025, so a 2024 week contains zero calls. The builder
    # must raise ValueError rather than silently return an empty instance.
    with pytest.raises(ValueError):
        build_weekly_instance(df, "2024-01-01", week_days=7)


# --- schedule + KPIs --------------------------------------------------------

def test_schedule_and_kpis_on_handmade_solution() -> None:
    """Given starts + assignment, schedule rows and KPIs are computed correctly."""
    # ARRANGE. 3 vessels, 2 berths. Vessel 0 is a service with a no-wait window
    # at t=0. We hand-build a KNOWN solution so the expected KPIs are easy to
    # compute by hand and check below.
    # Two berths, both able to serve type "X".
    berths = [Berth("B0", frozenset({"X"})), Berth("B1", frozenset({"X"}))]
    # `np.ones((3, 2), bool)` is a 3x2 all-True matrix: every vessel may use either
    # berth (no compatibility restriction in this test).
    compat = np.ones((3, 2), dtype=bool)
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
    # Local import to avoid a circular import at module load time.
    from bap_optim.weekly_instance import WeeklyInstanceBundle

    # Wrap the instance in the metadata bundle that the schedule/KPI helpers read.
    bundle = WeeklyInstanceBundle(
        instance=inst, berths=berths,
        vessel_ids=[0, 1, 2], vessel_names=["a", "b", "c"],
        vessel_types=["X", "X", "X"],
        arrivals_h=inst.arrivals, tau_h=np.array([10.0, 5.0, 6.0], np.float32),
        weights=inst.weights, is_service=inst.service,
        latest_start_h=inst.latest_start, week_start="w", week_end="w", source="test",
    )
    # The hand-made SOLUTION we are scoring:
    #   v0 -> B0 starting at t=0 (service, starts on arrival, no wait),
    #   v1 -> B1 starting at t=1, v2 -> B0 starting at t=10 (after v0 frees B0).
    starts = np.array([0.0, 1.0, 10.0], np.float32)
    # assignment[i] is a one-hot row: a 1 in the column of the berth vessel i uses.
    # Rows: v0->B0 (col 0), v1->B1 (col 1), v2->B0 (col 0).
    assignment = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)

    # ACT + ASSERT (schedule). One row per vessel.
    rows = assemble_schedule(bundle, starts, assignment)
    assert len(rows) == 3
    # ACT + ASSERT (KPIs), scored over a 24h horizon.
    kpis = compute_kpis(bundle, starts, assignment, horizon_h=24.0)
    # The lone service (v0) started exactly on arrival, so it never waited.
    # `is True` checks the value is the actual boolean True (not just truthy).
    assert kpis["all_services_no_wait"] is True
    assert kpis["service_max_wait_h"] == pytest.approx(0.0)
    # No vessel started after its latest-start window.
    assert kpis["window_violations"] == 0
    # makespan = last finish time = v2's finish = start 10 + tau 6 = 16.
    assert kpis["makespan_h"] == pytest.approx(16.0)  # v2 finishes at 10+6
    # B0 hosted v0 (tau 10) + v2 (tau 6) = 16h busy; B1 hosted v1 (tau 5) = 5h.
    # Utilization = busy hours / 24h horizon.
    assert kpis["berth_utilization"]["B0"] == pytest.approx(16.0 / 24.0)
    assert kpis["berth_utilization"]["B1"] == pytest.approx(5.0 / 24.0)

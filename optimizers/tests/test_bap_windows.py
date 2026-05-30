"""Solver-dependent tests for the extended DiscreteBAP: vessel-berth
compatibility and hard/soft no-wait service windows.

These need PyEPO + a working Gurobi (Pyomo) backend. They skip gracefully when
either is unavailable, so the suite still collects on machines without the
solver stack (unlike the legacy ``test_discrete_bap.py``, which hard-imports).
"""

import numpy as np
import pytest

# pytest.importorskip(...) tries to import "pyepo"; if it is not installed the
# WHOLE module is skipped (pytest stops collecting tests here) instead of
# erroring out. This is why the suite still "collects" on machines without the
# heavy solver stack -- see the module docstring above.
pytest.importorskip("pyepo")

# `# noqa: E402` silences the linter warning "import not at top of file": we
# deliberately import below the importorskip guard so the skip happens first.
from pyomo.environ import SolverFactory  # noqa: E402


# A plain helper (the leading underscore is a Python convention meaning
# "private / internal, not part of the public API"). The `-> bool` is a type
# hint saying this function returns a True/False value.
def _gurobi_available() -> bool:
    try:
        # Ask Pyomo whether the Gurobi commercial solver can actually be used.
        return bool(SolverFactory("gurobi").available())
    except Exception:
        # Any failure while probing the solver is treated as "not available".
        return False


# `pytestmark` is a special module-level name: pytest applies this mark to
# EVERY test in this file. @pytest.mark.skipif(condition, reason=...) skips the
# tests when `condition` is True -- here, when Gurobi is missing. So this whole
# file is double-guarded: pyepo must import AND Gurobi must be usable.
pytestmark = pytest.mark.skipif(not _gurobi_available(), reason="Gurobi solver not available")

from bap_optim.berths import DEFAULT_BERTHS, berth_names, vessel_berth_compat  # noqa: E402
from bap_optim.discrete_bap import DiscreteBAP, extract_decision  # noqa: E402
from bap_optim.instance import BAPInstance  # noqa: E402
from bap_optim.schedule import assemble_schedule, compute_kpis  # noqa: E402
from bap_optim.weekly_instance import generate_synthetic_weekly_instance  # noqa: E402


# Another private helper. `n: int = 3` and `hard: bool = True` are parameters
# with type hints AND default values, so callers can omit them.
# NOTE (review): the `hard` parameter is never used inside this function -- the
# hardness is actually decided by the caller via DiscreteBAP(hard_windows=...).
# It is harmless but dead; see REPORTED note below.
def _liquid_overbook_instance(n: int = 3, hard: bool = True) -> BAPInstance:
    """n liquid-bulk service vessels, all arriving at t=0 with a no-wait window,
    but only one berth (QC) can serve liquid bulk -> infeasible under hard windows.
    """
    # Build a compatibility matrix: all `n` vessels carry "Liquid Bulk", and only
    # the berth(s) able to handle liquid bulk will be marked compatible.
    compat = vessel_berth_compat(["Liquid Bulk"] * n, DEFAULT_BERTHS)
    return BAPInstance(
        n_vessels=n,
        n_berths=len(DEFAULT_BERTHS),
        arrivals=np.zeros(n, np.float32),      # all vessels arrive at time 0
        weights=np.ones(n, np.float32),
        big_m=500.0,
        latest_start=np.zeros(n, np.float32),  # no-wait: latest allowed start == 0
        berth_compat=compat,
        service=np.ones(n, bool),              # mark every vessel as a "service" vessel
    )


# Any function named `test_*` is auto-discovered and run by pytest. `-> None`
# just means it returns nothing (tests pass by not raising, fail on a bad
# assert). The tests below follow the ARRANGE (build inputs) -> ACT (solve) ->
# ASSERT (check results) structure.
def test_assignments_respect_compatibility() -> None:
    """No vessel is ever assigned to an incompatible berth."""
    # ARRANGE: build a reproducible synthetic week (seed=3 -> deterministic).
    bundle = generate_synthetic_weekly_instance(n_vessels=12, n_services=2, seed=3)
    # ACT: set the objective (per-vessel service times) and solve the model.
    model = DiscreteBAP(bundle.instance, hard_windows=True)
    model.setObj(bundle.tau_h)
    model.solve()
    # extract_decision returns (assignment-matrix, ...); `_` discards the part
    # we don't need. assign[i] is a one-hot row picking a berth for vessel i.
    assign, _ = extract_decision(model)
    # ASSERT: check the invariant for every vessel. A bare `assert <expr>` fails
    # the test when <expr> is False; the text after the comma is the failure msg.
    for i in range(bundle.n_vessels):
        b = int(np.argmax(assign[i]))  # the berth this vessel was assigned to
        # The f"..." f-string interpolates i and b into the failure message.
        assert bundle.instance.compatible(i, b), f"vessel {i} on incompatible berth {b}"
        # Exactly one berth chosen. pytest.approx compares floats with tolerance
        # (so 0.9999999 == 1.0 passes despite solver rounding).
        assert assign[i].sum() == pytest.approx(1.0)


def test_service_vessels_have_no_wait_hard() -> None:
    """With slack=0 and hard windows, service vessels start exactly at arrival."""
    # ARRANGE: same week, but service_slack_hours=0.0 forces a strict no-wait
    # window (the service must start the instant the vessel arrives).
    bundle = generate_synthetic_weekly_instance(n_vessels=12, n_services=2, seed=3,
                                                service_slack_hours=0.0)
    # ACT: solve; `starts[i]` is the chosen start time (hours) for vessel i.
    model = DiscreteBAP(bundle.instance, hard_windows=True)
    model.setObj(bundle.tau_h)
    starts, _ = model.solve()
    assign, _ = extract_decision(model)
    # Roll the solution up into key performance indicators over a 168h (1-week) horizon.
    kpis = compute_kpis(bundle, starts, assign, horizon_h=168.0)
    # ASSERT: `is True` checks the value is literally the boolean True (not just
    # truthy), matching the docstring's "all services no-wait" intent.
    assert kpis["all_services_no_wait"] is True
    assert kpis["window_violations"] == 0
    # np.flatnonzero gives the integer indices where bundle.is_service is True;
    # for each such service vessel its start must equal its arrival (within
    # abs=1e-2 hours -> ~36 seconds of float tolerance).
    for i in np.flatnonzero(bundle.is_service):
        assert starts[i] == pytest.approx(bundle.arrivals_h[i], abs=1e-2)


def test_hard_overbooked_services_infeasible() -> None:
    """Three no-wait liquid services + one liquid berth -> infeasible (raises)."""
    # ARRANGE: 3 vessels that all must start at t=0 but only one berth can serve
    # them -> physically impossible under hard windows.
    inst = _liquid_overbook_instance(n=3, hard=True)
    model = DiscreteBAP(inst, hard_windows=True)
    model.setObj(np.full(3, 10.0, np.float32))  # np.full -> array [10, 10, 10]
    # pytest.raises asserts that the block raises the given exception. The test
    # PASSES only if model.solve() raises RuntimeError (i.e. the solver reports
    # the problem as infeasible); it FAILS if no error is raised.
    with pytest.raises(RuntimeError):
        model.solve()


def test_soft_overbooked_services_feasible_with_tardiness() -> None:
    """Same over-booked case stays feasible under soft windows, with tardiness."""
    # ARRANGE: same impossible-under-hard scenario, but hard_windows=False turns
    # the window into a soft constraint: lateness is allowed but penalized.
    inst = _liquid_overbook_instance(n=3)
    model = DiscreteBAP(inst, hard_windows=False, penalty_weight=1000.0)
    model.setObj(np.full(3, 10.0, np.float32))
    starts, _ = model.solve()  # now solvable -> returns start times instead of raising
    # 3 vessels stacked on one berth, tau=10 each -> starts 0,10,20 -> tardiness 30.
    # tardiness = sum of how late each start is past its latest_start (clamped at
    # 0 by np.maximum so early/on-time starts contribute nothing).
    tardiness = float(np.sum(np.maximum(0.0, starts - inst.latest_start)))
    assert tardiness > 1e-2  # there MUST be some lateness in this over-booked case
    # The three vessels must be sequenced back-to-back on the single berth.
    # sorted(...) makes the check order-independent (any permutation of vessels).
    assert sorted(np.round(starts, 1)) == [0.0, 10.0, 20.0]


def test_backward_compatible_no_windows_no_compat() -> None:
    """An instance with no windows/compat behaves like the original model."""
    # Local import inside the test: the legacy generator lives in discrete_bap;
    # importing here keeps it scoped to this single test.
    from bap_optim.discrete_bap import generate_bap_instance

    # ARRANGE: a plain instance with no compatibility matrix and no service flags.
    inst = generate_bap_instance(n_vessels=6, n_berths=2, horizon_hours=80.0, seed=0)
    # Sanity-check the fixture: these new optional fields really are absent.
    assert inst.berth_compat is None and inst.service is None
    model = DiscreteBAP(inst)  # default hard_windows=True, but no service vessels
    model.setObj(np.array([12, 8, 20, 5, 15, 10], np.float32))
    starts, obj = model.solve()
    assign, _ = extract_decision(model)
    # ASSERT: one start time per vessel.
    assert starts.shape == (6,)
    # No vessel starts before it arrives (minus 1e-3 float slack). `.all()`
    # requires the condition to hold for every element of the boolean array.
    assert (starts >= inst.arrivals - 1e-3).all()
    # Every vessel assigned to exactly one berth. np.allclose is the array-wide
    # float-tolerant equality (each row-sum ~= 1.0 within atol=1e-3).
    assert np.allclose(assign.sum(axis=1), 1.0, atol=1e-3)
    assert obj > 0  # a real, positive objective value was returned

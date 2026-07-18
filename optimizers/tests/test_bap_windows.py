"""Solver-dependent tests for the extended DiscreteBAP: vessel-berth
compatibility and hard/soft no-wait service windows.

These need PyEPO + a working Gurobi (Pyomo) backend. They skip gracefully when
either is unavailable, so the suite still collects on machines without the
solver stack (unlike the legacy ``test_discrete_bap.py``, which hard-imports).
"""

import numpy as np
import pytest

# Skip the whole module if pyepo is missing, so the suite still collects without
# the heavy solver stack (unlike the legacy test_discrete_bap.py).
pytest.importorskip("pyepo")

from pyomo.environ import SolverFactory  # noqa: E402


def _gurobi_available() -> bool:
    try:
        return bool(SolverFactory("gurobi").available())
    except Exception:
        return False


# Double-guard: pyepo must import AND Gurobi must be usable.
pytestmark = pytest.mark.skipif(not _gurobi_available(), reason="Gurobi solver not available")

from bap_optim.berths import DEFAULT_BERTHS, berth_names, vessel_berth_compat  # noqa: E402
from bap_optim.discrete_bap import DiscreteBAP, extract_decision  # noqa: E402
from bap_optim.instance import BAPInstance  # noqa: E402
from bap_optim.schedule import assemble_schedule, compute_kpis  # noqa: E402
from bap_optim.weekly_instance import generate_synthetic_weekly_instance  # noqa: E402


# NOTE (review): the `hard` parameter is never used -- hardness is decided by the
# caller via DiscreteBAP(hard_windows=...). Harmless but dead; see REPORTED note.
def _liquid_overbook_instance(n: int = 3, hard: bool = True) -> BAPInstance:
    """n liquid-bulk service vessels, all arriving at t=0 with a no-wait window,
    but only one berth (QC) can serve liquid bulk -> infeasible under hard windows.
    """
    compat = vessel_berth_compat(["Liquid Bulk"] * n, DEFAULT_BERTHS)
    return BAPInstance(
        n_vessels=n,
        n_berths=len(DEFAULT_BERTHS),
        arrivals=np.zeros(n, np.float32),
        weights=np.ones(n, np.float32),
        big_m=500.0,
        latest_start=np.zeros(n, np.float32),  # no-wait: latest allowed start == 0
        berth_compat=compat,
        service=np.ones(n, bool),
    )


def test_assignments_respect_compatibility() -> None:
    """No vessel is ever assigned to an incompatible berth."""
    bundle = generate_synthetic_weekly_instance(n_vessels=12, n_services=2, seed=3)
    model = DiscreteBAP(bundle.instance, hard_windows=True)
    model.setObj(bundle.tau_h)
    model.solve()
    assign, _ = extract_decision(model)
    for i in range(bundle.n_vessels):
        b = int(np.argmax(assign[i]))
        assert bundle.instance.compatible(i, b), f"vessel {i} on incompatible berth {b}"
        # Exactly one berth chosen (approx absorbs solver rounding like 0.9999999).
        assert assign[i].sum() == pytest.approx(1.0)


def test_service_vessels_have_no_wait_hard() -> None:
    """With slack=0 and hard windows, service vessels start exactly at arrival."""
    # service_slack_hours=0.0 forces a strict no-wait window.
    bundle = generate_synthetic_weekly_instance(n_vessels=12, n_services=2, seed=3,
                                                service_slack_hours=0.0)
    model = DiscreteBAP(bundle.instance, hard_windows=True)
    model.setObj(bundle.tau_h)
    starts, _ = model.solve()
    assign, _ = extract_decision(model)
    kpis = compute_kpis(bundle, starts, assign, horizon_h=168.0)
    assert kpis["all_services_no_wait"] is True
    assert kpis["window_violations"] == 0
    # Each service vessel's start must equal its arrival (~36s float tolerance).
    for i in np.flatnonzero(bundle.is_service):
        assert starts[i] == pytest.approx(bundle.arrivals_h[i], abs=1e-2)


def test_hard_overbooked_services_infeasible() -> None:
    """Three no-wait liquid services + one liquid berth -> infeasible (raises)."""
    inst = _liquid_overbook_instance(n=3, hard=True)
    model = DiscreteBAP(inst, hard_windows=True)
    model.setObj(np.full(3, 10.0, np.float32))
    # solve() must raise RuntimeError (solver reports infeasible).
    with pytest.raises(RuntimeError):
        model.solve()


def test_soft_overbooked_services_feasible_with_tardiness() -> None:
    """Same over-booked case stays feasible under soft windows, with tardiness."""
    # hard_windows=False turns the window into a soft (penalized) constraint.
    inst = _liquid_overbook_instance(n=3)
    model = DiscreteBAP(inst, hard_windows=False, penalty_weight=1000.0)
    model.setObj(np.full(3, 10.0, np.float32))
    starts, _ = model.solve()
    # 3 vessels stacked on one berth, tau=10 each -> starts 0,10,20 -> tardiness 30.
    tardiness = float(np.sum(np.maximum(0.0, starts - inst.latest_start)))
    assert tardiness > 1e-2  # there MUST be lateness in this over-booked case
    # The three vessels sequence back-to-back on the single berth.
    assert sorted(np.round(starts, 1)) == [0.0, 10.0, 20.0]


def test_backward_compatible_no_windows_no_compat() -> None:
    """An instance with no windows/compat behaves like the original model."""
    from bap_optim.discrete_bap import generate_bap_instance

    inst = generate_bap_instance(n_vessels=6, n_berths=2, horizon_hours=80.0, seed=0)
    # The new optional fields really are absent.
    assert inst.berth_compat is None and inst.service is None
    model = DiscreteBAP(inst)
    model.setObj(np.array([12, 8, 20, 5, 15, 10], np.float32))
    starts, obj = model.solve()
    assign, _ = extract_decision(model)
    assert starts.shape == (6,)
    # No vessel starts before it arrives (1e-3 float slack).
    assert (starts >= inst.arrivals - 1e-3).all()
    # Every vessel assigned to exactly one berth (float-tolerant row-sums).
    assert np.allclose(assign.sum(axis=1), 1.0, atol=1e-3)
    # Total waiting is non-negative. (It is exactly 0 for this instance: the six
    # arrivals are spread widely enough that alternating berths gives a
    # conflict-free schedule, so `> 0` was a wrong assumption about contention.)
    assert obj >= -1e-3

"""Solver-dependent tests for the shared navigation-channel resource.

Skips gracefully without PyEPO + a working Gurobi backend.
"""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("pyepo")

from pyomo.environ import SolverFactory  # noqa: E402


def _gurobi_available() -> bool:
    try:
        return bool(SolverFactory("gurobi").available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gurobi_available(), reason="Gurobi solver not available")

from bap_optim.discrete_bap import (  # noqa: E402
    DiscreteBAP,
    extract_channel,
    extract_decision,
    generate_bap_instance,
)

C = 2.0  # channel transit time (hours)


def _solve_with_channel(n_vessels=5, n_berths=2, seed=0, objective="waiting"):
    inst = generate_bap_instance(n_vessels=n_vessels, n_berths=n_berths,
                                 horizon_hours=80.0, seed=seed)
    inst = dataclasses.replace(inst, channel_time=C)
    rng = np.random.default_rng(seed)
    tau = rng.uniform(8.0, 25.0, size=n_vessels).astype(np.float32)
    model = DiscreteBAP(inst, objective=objective)
    model.setObj(tau)
    starts, obj = model.solve()
    ein, eout = extract_channel(model)
    return inst, tau, starts, obj, ein, eout, model


def test_channel_transits_never_overlap() -> None:
    """No two channel transits (entry or exit, any vessels) overlap."""
    inst, tau, starts, obj, ein, eout, _ = _solve_with_channel()
    intervals = [(float(ein[i]), float(ein[i]) + C) for i in range(inst.n_vessels)]
    intervals += [(float(eout[i]), float(eout[i]) + C) for i in range(inst.n_vessels)]
    intervals.sort()
    for (s0, e0), (s1, e1) in zip(intervals, intervals[1:]):
        assert s1 >= e0 - 1e-2, f"channel overlap: [{s0:.2f},{e0:.2f}] vs [{s1:.2f},{e1:.2f}]"


def test_channel_links_hold() -> None:
    """Berth start follows the inbound transit; exit follows service."""
    inst, tau, starts, obj, ein, eout, _ = _solve_with_channel()
    for i in range(inst.n_vessels):
        assert starts[i] >= ein[i] + C - 1e-2          # moor only after entering
        assert ein[i] >= inst.arrivals[i] - 1e-2       # enter only after arrival
        assert eout[i] >= starts[i] + tau[i] - 1e-2    # exit only after service


def test_channel_waiting_objective() -> None:
    """The default 'waiting' objective equals Σᵢ (einᵢ − aᵢ) when a channel is modelled
    (vessels wait at anchorage until they may enter the channel)."""
    inst, tau, starts, obj, ein, eout, _ = _solve_with_channel(objective="waiting")
    expected = float(np.sum(ein - inst.arrivals))
    assert obj == pytest.approx(expected, rel=1e-3, abs=1e-2)


def test_channel_idle_objective() -> None:
    """The 'idle' objective equals the summed per-berth occupied span
    Σ_b (max completion − min start) over vessels assigned to b."""
    inst, tau, starts, obj, ein, eout, model = _solve_with_channel(objective="idle")
    assign, _ = extract_decision(model)
    span = 0.0
    for b in range(inst.n_berths):
        at_b = [i for i in range(inst.n_vessels) if assign[i, b] > 0.5]
        if not at_b:
            continue
        span += max(starts[i] + tau[i] for i in at_b) - min(starts[i] for i in at_b)
    assert obj == pytest.approx(span, rel=1e-3, abs=1e-2)


def test_no_channel_waiting_objective() -> None:
    """With no channel the default 'waiting' objective is Σᵢ (sᵢ − aᵢ) and no channel
    vars are added (entry collapses to the berth start)."""
    inst = generate_bap_instance(n_vessels=5, n_berths=2, horizon_hours=80.0, seed=0)
    assert inst.channel_time is None
    model = DiscreteBAP(inst)  # default objective="waiting"
    tau = np.array([12, 8, 20, 5, 15], dtype=np.float32)
    model.setObj(tau)
    starts, obj = model.solve()
    assert extract_channel(model) == (None, None)
    assert obj == pytest.approx(float(np.sum(starts - inst.arrivals)), rel=1e-3, abs=1e-2)

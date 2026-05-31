"""Tests for the discrete BAP MILP and decision-quality utilities."""

import numpy as np
import pytest

# NOTE: hard-imports Pyomo/PyEPO/Gurobi via discrete_bap; collection fails if the
# solver stack is missing (see review note re: guarding with pytest.importorskip).
from bap_optim.discrete_bap import (
    DiscreteBAP,
    derive_starts_under_true_tau,
    extract_decision,
    generate_bap_instance,
    schedule_cost_under_true_tau,
)


@pytest.fixture
def small_instance():
    # Fixed seed -> deterministic instance. 5 vessels, 2 berths, 80h horizon.
    return generate_bap_instance(n_vessels=5, n_berths=2, horizon_hours=80.0, seed=0)


def test_dbap_solves_to_feasible(small_instance) -> None:
    """A DBAP solve produces feasible start times respecting arrivals."""
    model = DiscreteBAP(small_instance)
    tau = np.array([15.0, 20.0, 25.0, 18.0, 22.0], dtype=np.float32)
    model.setObj(tau)
    starts, obj = model.solve()
    # Every start must be ≥ corresponding arrival (1e-3 float tolerance).
    assert (starts >= small_instance.arrivals - 1e-3).all()
    # Weighted-completion-time objective is strictly positive here.
    assert obj > 0


def test_extract_decision_assignment_sums_to_one(small_instance) -> None:
    """Each vessel must be assigned to exactly one berth."""
    model = DiscreteBAP(small_instance)
    tau = np.full(small_instance.n_vessels, 20.0, dtype=np.float32)
    model.setObj(tau)
    model.solve()
    assign, _ = extract_decision(model)
    # Each vessel goes to exactly one berth, so every row-sum must equal 1
    # (float-tolerant: the binaries come back as floats like 0.9999).
    np.testing.assert_allclose(assign.sum(axis=1), 1.0, atol=1e-3)


def test_derive_starts_respects_precedence(small_instance) -> None:
    """Re-derived starts must be feasible: arrivals + precedence."""
    model = DiscreteBAP(small_instance)
    tau = np.array([15.0, 20.0, 25.0, 18.0, 22.0], dtype=np.float32)
    model.setObj(tau)
    model.solve()
    assign, order = extract_decision(model)
    # Recompute feasible starts for that decision under the same τ — this is the
    # helper regret relies on, so we verify it yields a feasible schedule.
    starts = derive_starts_under_true_tau(assign, order, tau, small_instance.arrivals)
    # Feasibility 1: nobody starts before they arrive (1e-3 float slack).
    assert (starts >= small_instance.arrivals - 1e-3).all()
    # Feasibility 2: within each berth, no two vessels overlap.
    for b in range(small_instance.n_berths):
        at_b = [i for i in range(small_instance.n_vessels) if assign[i, b] > 0.5]
        at_b.sort(key=lambda i: starts[i])
        # Each later vessel must not start until the earlier one finishes.
        for prev, nxt in zip(at_b, at_b[1:]):
            assert starts[nxt] >= starts[prev] + tau[prev] - 1e-3


def test_fi_decision_is_minimum_cost(small_instance) -> None:
    """The full-information (FI) decision under true τ should have the smallest
    feasible cost among a set of alternatives produced from random cost vectors.
    """
    model = DiscreteBAP(small_instance)
    true_tau = np.array([15.0, 20.0, 25.0, 18.0, 22.0], dtype=np.float32)

    # Full-information benchmark: solve under the true τ and score under true τ.
    # By construction this is the best achievable cost.
    model.setObj(true_tau)
    model.solve()
    assign_fi, order_fi = extract_decision(model)
    cost_fi, _ = schedule_cost_under_true_tau(
        assign_fi, order_fi, true_tau, small_instance.arrivals, small_instance.weights
    )

    rng = np.random.default_rng(0)
    for _ in range(5):
        # Decision driven by a WRONG τ; when both are scored against the SAME
        # true τ it cannot beat the full-information decision (property under test).
        bad_tau = rng.uniform(5, 50, size=small_instance.n_vessels).astype(np.float32)
        model.setObj(bad_tau)
        model.solve()
        a, o = extract_decision(model)
        cost, _ = schedule_cost_under_true_tau(
            a, o, true_tau, small_instance.arrivals, small_instance.weights
        )
        # The -5.0 absolute slack absorbs the 0.5% MIP gap (a near-optimal solve
        # can be slightly cheaper than the FI solve).
        assert cost >= cost_fi - 5.0  # within solver gap


def test_regret_is_nonnegative(small_instance) -> None:
    """Regret of any prediction-driven decision is ≥ 0 modulo solver gap."""
    model = DiscreteBAP(small_instance)
    true_tau = np.array([15.0, 20.0, 25.0, 18.0, 22.0], dtype=np.float32)
    rng = np.random.default_rng(1)
    # A single predicted τ̂ standing in for a model's (imperfect) output.
    pred_tau = rng.uniform(5, 50, size=5).astype(np.float32)

    # Decision made under the PREDICTION, then scored against the true τ.
    model.setObj(pred_tau)
    model.solve()
    a_p, o_p = extract_decision(model)
    cost_pred, _ = schedule_cost_under_true_tau(
        a_p, o_p, true_tau, small_instance.arrivals, small_instance.weights
    )

    # Full-information decision (solve under true τ), scored against true τ.
    model.setObj(true_tau)
    model.solve()
    a_fi, o_fi = extract_decision(model)
    cost_fi, _ = schedule_cost_under_true_tau(
        a_fi, o_fi, true_tau, small_instance.arrivals, small_instance.weights
    )

    # Regret = cost(predicted) − cost(FI) must be ≥ 0 by definition (FI is optimal
    # under true τ). The -5.0 tolerance absorbs the solver's 0.5% MIP gap.
    assert cost_pred - cost_fi >= -5.0  # solver gap allowance

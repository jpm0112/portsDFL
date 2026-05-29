"""Tests for the discrete BAP MILP and decision-quality utilities."""

import numpy as np
import pytest

from ports_dfl.optim.discrete_bap import (
    DiscreteBAP,
    derive_starts_under_true_tau,
    extract_decision,
    generate_bap_instance,
    schedule_cost_under_true_tau,
)


@pytest.fixture
def small_instance():
    return generate_bap_instance(n_vessels=5, n_berths=2, horizon_hours=80.0, seed=0)


def test_dbap_solves_to_feasible(small_instance) -> None:
    """A DBAP solve produces feasible start times respecting arrivals."""
    model = DiscreteBAP(small_instance)
    tau = np.array([15.0, 20.0, 25.0, 18.0, 22.0], dtype=np.float32)
    model.setObj(tau)
    starts, obj = model.solve()
    # Every start must be ≥ corresponding arrival
    assert (starts >= small_instance.arrivals - 1e-3).all()
    assert obj > 0


def test_extract_decision_assignment_sums_to_one(small_instance) -> None:
    """Each vessel must be assigned to exactly one berth."""
    model = DiscreteBAP(small_instance)
    tau = np.full(small_instance.n_vessels, 20.0, dtype=np.float32)
    model.setObj(tau)
    model.solve()
    assign, _ = extract_decision(model)
    np.testing.assert_allclose(assign.sum(axis=1), 1.0, atol=1e-3)


def test_derive_starts_respects_precedence(small_instance) -> None:
    """Re-derived starts must be feasible: arrivals + precedence."""
    model = DiscreteBAP(small_instance)
    tau = np.array([15.0, 20.0, 25.0, 18.0, 22.0], dtype=np.float32)
    model.setObj(tau)
    model.solve()
    assign, order = extract_decision(model)
    starts = derive_starts_under_true_tau(assign, order, tau, small_instance.arrivals)
    assert (starts >= small_instance.arrivals - 1e-3).all()
    # Per-berth: start of next ≥ completion of previous
    for b in range(small_instance.n_berths):
        at_b = [i for i in range(small_instance.n_vessels) if assign[i, b] > 0.5]
        # Sort by start
        at_b.sort(key=lambda i: starts[i])
        for prev, nxt in zip(at_b, at_b[1:]):
            assert starts[nxt] >= starts[prev] + tau[prev] - 1e-3


def test_fi_decision_is_minimum_cost(small_instance) -> None:
    """The full-information (FI) decision under true τ should have the smallest
    feasible cost among a set of alternatives produced from random cost vectors.
    """
    model = DiscreteBAP(small_instance)
    true_tau = np.array([15.0, 20.0, 25.0, 18.0, 22.0], dtype=np.float32)

    # Full-information benchmark: solve under true τ
    model.setObj(true_tau)
    model.solve()
    assign_fi, order_fi = extract_decision(model)
    cost_fi, _ = schedule_cost_under_true_tau(
        assign_fi, order_fi, true_tau, small_instance.arrivals, small_instance.weights
    )

    rng = np.random.default_rng(0)
    for _ in range(5):
        bad_tau = rng.uniform(5, 50, size=small_instance.n_vessels).astype(np.float32)
        model.setObj(bad_tau)
        model.solve()
        a, o = extract_decision(model)
        cost, _ = schedule_cost_under_true_tau(
            a, o, true_tau, small_instance.arrivals, small_instance.weights
        )
        assert cost >= cost_fi - 5.0  # within solver gap


def test_regret_is_nonnegative(small_instance) -> None:
    """Regret of any prediction-driven decision is ≥ 0 modulo solver gap."""
    model = DiscreteBAP(small_instance)
    true_tau = np.array([15.0, 20.0, 25.0, 18.0, 22.0], dtype=np.float32)
    rng = np.random.default_rng(1)
    pred_tau = rng.uniform(5, 50, size=5).astype(np.float32)

    model.setObj(pred_tau)
    model.solve()
    a_p, o_p = extract_decision(model)
    cost_pred, _ = schedule_cost_under_true_tau(
        a_p, o_p, true_tau, small_instance.arrivals, small_instance.weights
    )

    model.setObj(true_tau)
    model.solve()
    a_fi, o_fi = extract_decision(model)
    cost_fi, _ = schedule_cost_under_true_tau(
        a_fi, o_fi, true_tau, small_instance.arrivals, small_instance.weights
    )

    assert cost_pred - cost_fi >= -5.0  # solver gap allowance

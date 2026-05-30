"""Tests for the discrete BAP MILP and decision-quality utilities."""

# This is a pytest test file. pytest "collects" (auto-discovers) every function
# whose name starts with `test_` and runs it as a test. A test PASSES if it
# returns without error and FAILS if any `assert` inside it is False or it raises.

import numpy as np
import pytest

# Import the optimizer and helper functions under test. NOTE: this import line
# pulls in Pyomo/PyEPO/Gurobi (via discrete_bap). If those solver libraries are
# not installed, collection of this whole file fails — see the review note about
# guarding with pytest.importorskip.
from ports_dfl.optim.discrete_bap import (
    DiscreteBAP,
    derive_starts_under_true_tau,
    extract_decision,
    generate_bap_instance,
    schedule_cost_under_true_tau,
)


# `@pytest.fixture` marks a reusable piece of setup. A test "requests" this
# fixture simply by listing its name (`small_instance`) as a parameter; pytest
# then calls this function and passes the return value in. This keeps each test
# from rebuilding the same instance by hand.
@pytest.fixture
def small_instance():
    # A fixed `seed=0` makes the synthetic instance deterministic, so the tests
    # are reproducible run-to-run. 5 vessels, 2 berths over an 80h horizon.
    return generate_bap_instance(n_vessels=5, n_berths=2, horizon_hours=80.0, seed=0)


# A test function: pytest discovers it by the `test_` prefix and runs it.
# `small_instance` as a parameter means pytest injects the fixture above.
# `-> None` just documents that the test returns nothing.
def test_dbap_solves_to_feasible(small_instance) -> None:
    """A DBAP solve produces feasible start times respecting arrivals."""
    # ARRANGE: build the optimizer and a known service-time vector τ (one entry
    # per vessel). These tests follow the arrange–act–assert shape: set things
    # up, run the operation, then check the result.
    model = DiscreteBAP(small_instance)
    tau = np.array([15.0, 20.0, 25.0, 18.0, 22.0], dtype=np.float32)
    # ACT: load τ into the model and solve the MILP for start times + objective.
    model.setObj(tau)
    starts, obj = model.solve()
    # ASSERT: `assert <expr>` fails the test if <expr> is False. Here the
    # comparison is element-wise (numpy), and `.all()` collapses it to a single
    # True only if EVERY vessel's start is ≥ its arrival (with a 1e-3 float
    # tolerance, since solver outputs are floats and never exactly equal).
    # Every start must be ≥ corresponding arrival
    assert (starts >= small_instance.arrivals - 1e-3).all()
    # A correct weighted-completion-time objective is strictly positive here
    # (positive weights, positive completion times).
    assert obj > 0


def test_extract_decision_assignment_sums_to_one(small_instance) -> None:
    """Each vessel must be assigned to exactly one berth."""
    model = DiscreteBAP(small_instance)
    # `np.full(n, 20.0)` makes a length-n array filled with 20.0 — a uniform τ.
    tau = np.full(small_instance.n_vessels, 20.0, dtype=np.float32)
    model.setObj(tau)
    model.solve()
    # `assign, _ = ...` unpacks the (assignment, order) tuple; the underscore is
    # a convention for "value we deliberately ignore" (we don't need the order).
    assign, _ = extract_decision(model)
    # assign has shape (N vessels, B berths). `assign.sum(axis=1)` sums across
    # berths for each vessel, giving one number per vessel. Encodes the MILP
    # rule "each vessel goes to exactly one berth", so every entry must equal 1.
    # `np.testing.assert_allclose` is float-tolerant equality (raises/fails if
    # any element differs by more than atol); used because the binaries come
    # back as floats like 0.9999.
    np.testing.assert_allclose(assign.sum(axis=1), 1.0, atol=1e-3)


def test_derive_starts_respects_precedence(small_instance) -> None:
    """Re-derived starts must be feasible: arrivals + precedence."""
    model = DiscreteBAP(small_instance)
    tau = np.array([15.0, 20.0, 25.0, 18.0, 22.0], dtype=np.float32)
    model.setObj(tau)
    model.solve()
    # Pull the locked-in decision (which berth + ordering) out of the solved model.
    assign, order = extract_decision(model)
    # Recompute feasible start times for that decision under the same τ. This is
    # the helper regret relies on, so we verify it produces a feasible schedule.
    starts = derive_starts_under_true_tau(assign, order, tau, small_instance.arrivals)
    # Feasibility check 1: nobody starts before they arrive (1e-3 float slack).
    assert (starts >= small_instance.arrivals - 1e-3).all()
    # Feasibility check 2: within each berth, no two vessels overlap.
    # Per-berth: start of next ≥ completion of previous
    for b in range(small_instance.n_berths):
        # Vessels assigned to berth b (binary stored as float, so test > 0.5).
        at_b = [i for i in range(small_instance.n_vessels) if assign[i, b] > 0.5]
        # Sort that berth's vessels by start time. `key=lambda i: starts[i]` is a
        # tiny inline function giving the sort key (the start time of vessel i) —
        # this mirrors how derive_starts_under_true_tau packs them in order.
        # Sort by start
        at_b.sort(key=lambda i: starts[i])
        # `zip(at_b, at_b[1:])` pairs each vessel with the one after it
        # (consecutive pairs). For each adjacent pair, the later vessel must not
        # start until the earlier one has finished (start + its service time),
        # i.e. no double-booking the berth. Minus 1e-3 = float tolerance.
        for prev, nxt in zip(at_b, at_b[1:]):
            assert starts[nxt] >= starts[prev] + tau[prev] - 1e-3


def test_fi_decision_is_minimum_cost(small_instance) -> None:
    """The full-information (FI) decision under true τ should have the smallest
    feasible cost among a set of alternatives produced from random cost vectors.
    """
    model = DiscreteBAP(small_instance)
    true_tau = np.array([15.0, 20.0, 25.0, 18.0, 22.0], dtype=np.float32)

    # Full-information (FI) benchmark: solve the MILP using the TRUE τ, then
    # score that decision under the true τ. By construction this is the best
    # achievable cost — every other decision below is compared against it.
    # Full-information benchmark: solve under true τ
    model.setObj(true_tau)
    model.solve()
    assign_fi, order_fi = extract_decision(model)
    cost_fi, _ = schedule_cost_under_true_tau(
        assign_fi, order_fi, true_tau, small_instance.arrivals, small_instance.weights
    )

    # Seeded RNG -> the same 5 "bad" τ vectors every run (reproducible test).
    rng = np.random.default_rng(0)
    # `for _ in range(5)`: repeat 5 times; `_` means we don't use the loop index.
    for _ in range(5):
        # A decision driven by a WRONG τ (random draw). Decisions made under bad
        # predictions can't beat the full-information decision when both are then
        # scored against the SAME true τ — that is the property under test.
        bad_tau = rng.uniform(5, 50, size=small_instance.n_vessels).astype(np.float32)
        model.setObj(bad_tau)
        model.solve()
        a, o = extract_decision(model)
        # Score this alternative decision under the TRUE τ (apples-to-apples).
        cost, _ = schedule_cost_under_true_tau(
            a, o, true_tau, small_instance.arrivals, small_instance.weights
        )
        # The alternative's realised cost must be ≥ the FI cost. The `- 5.0`
        # absolute slack absorbs the 0.5% MIP gap the solver is allowed (a
        # near-optimal solve can be slightly cheaper than the FI solve), so the
        # check stays robust without weakening the ≥ property.
        assert cost >= cost_fi - 5.0  # within solver gap


def test_regret_is_nonnegative(small_instance) -> None:
    """Regret of any prediction-driven decision is ≥ 0 modulo solver gap."""
    model = DiscreteBAP(small_instance)
    true_tau = np.array([15.0, 20.0, 25.0, 18.0, 22.0], dtype=np.float32)
    rng = np.random.default_rng(1)
    # A single predicted τ̂ standing in for a model's (imperfect) output.
    # `size=5` matches the fixture's 5 vessels.
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

    # Regret = cost(predicted decision) − cost(FI decision). It must be ≥ 0 by
    # definition (FI is optimal under true τ). The `-5.0` tolerance again absorbs
    # the solver's 0.5% MIP gap, so a tiny negative value isn't a real violation.
    assert cost_pred - cost_fi >= -5.0  # solver gap allowance

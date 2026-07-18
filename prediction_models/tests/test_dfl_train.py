"""Smoke regression test for the blackbox-DFL training loop.

The DFL trainers are the project's core contribution but had no direct test
(only the underlying MILP regret arithmetic was covered). This exercises the
whole loop end-to-end on a tiny seeded synthetic instance and pins the
invariants that the Round-2 fixes established: the loop runs, produces finite
traces, and regret stays >= 0 now that the loss/metric match the objective.

Requires the full solver + torch stack; skipped cleanly when any piece is
absent (torch/pyepo/gurobi are optional deps of this repo).
"""

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch required for DFL training")
pytest.importorskip("pyepo", reason="PyEPO required for DFL training")
pytest.importorskip("gurobipy", reason="Gurobi required to solve the MILP")
pytest.importorskip("bap_optim", reason="bap_optim package must be importable")

from bap_optim.classic_bap import make_classic_problem
from ports_dfl.models.linear import _LinearHead
from ports_dfl.train.dfl_blackbox import DFLBlackboxConfig, train_dfl_blackbox

SEED = 1048596


def _tiny_problem():
    """A small, high-contention synthetic problem (fast to solve)."""
    return make_classic_problem(
        n_vessels=4,
        n_berths=2,
        n_train=6,
        n_val=4,
        contention=1.0,
        weight_dist="three_class",
        arrival="uniform",
        tau_mean=10.0,
        tau_sigma=0.5,
        noise_std=0.4,
        n_noise_features=3,
        include_weight_feature=True,
        seed=SEED,
    )


def test_dfl_blackbox_loop_runs_and_regret_is_nonnegative():
    prob = _tiny_problem()
    n_features = prob.X_train.shape[-1]
    model = _LinearHead(n_features)
    cfg = DFLBlackboxConfig(
        lr=1e-3,
        weight_decay=0.0,
        batch_size=2,
        max_epochs=2,
        patience=2,
        blackbox_lambd=10.0,
        processes=1,
        seed=SEED,
    )

    result = train_dfl_blackbox(
        model,
        prob.X_train, prob.tau_train,
        prob.X_val, prob.tau_val,
        prob.instance,
        cfg,
    )

    # The loop actually ran and recorded a trace per epoch.
    assert result.epochs_run >= 1
    assert len(result.train_loss_history) >= 1
    assert len(result.val_regret_history) >= 1
    # Every recorded value is finite (no nan/inf leaking from an empty val set).
    assert all(np.isfinite(v) for v in result.train_loss_history)
    assert all(np.isfinite(v) for v in result.val_regret_history)
    assert np.isfinite(result.best_val_regret)
    # Regret >= 0 by construction now that the unweighted cost matches the MILP
    # objective (the 1e-3 slack only absorbs float32 reconstruction noise).
    assert min(result.val_regret_history) >= -1e-3
    assert result.best_val_regret >= -1e-3

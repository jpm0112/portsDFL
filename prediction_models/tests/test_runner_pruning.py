"""Tests for the opt-in per-fold pruning branch in tuning.runner.

These exercise ``make_objective(report_intermediate=...)`` together with
``_evaluate_one_config``'s per-fold trial reporting. The preprocessor is
monkeypatched to an identity passthrough so the test needs neither the data
layer nor the real feature schema, and a constant-prediction stub model makes
the resulting MAE fully controllable.
"""

import numpy as np
import optuna
import pandas as pd
import pytest

from ports_dfl.tuning import runner


class _ConstModel:
    """Minimal BaseModel-like stub: predicts a constant so MAE is controllable."""

    def __init__(self, input_dim=None, offset=0.0):
        self.offset = offset

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        return self

    def predict(self, X):
        return np.full(len(X), self.offset)


class _IdentityPre:
    """Passthrough preprocessor: returns the DataFrame's values unchanged."""

    def fit_transform(self, X, y=None):
        return X.to_numpy(dtype=float)

    def transform(self, X):
        return X.to_numpy(dtype=float)


@pytest.fixture
def toy_problem():
    """A tiny regression problem (true target 0) with 5 contiguous folds."""
    n = 50
    X = pd.DataFrame({"f": np.linspace(0.0, 1.0, n)})
    y = pd.Series(np.zeros(n))
    idx = np.arange(n)
    splits = [(np.delete(idx, chunk), chunk) for chunk in np.array_split(idx, 5)]
    return X, y, splits


@pytest.fixture(autouse=True)
def _identity_preprocessor(monkeypatch):
    # Replace the real preprocessor so the runner needs no data layer / schema.
    monkeypatch.setattr(runner, "build_preprocessor", lambda **kw: _IdentityPre())


def _const_objective(offset, X, y, splits, report_intermediate):
    """Objective over a constant-prediction model fixed at ``offset``."""

    def suggest(trial):
        return {"offset": trial.suggest_categorical("offset", [offset])}

    return runner.make_objective(
        factory=_ConstModel,
        suggest_fn=suggest,
        X=X,
        y=y,
        splits=splits,
        report_intermediate=report_intermediate,
    )


def test_report_intermediate_prunes_bad_config(toy_problem) -> None:
    """With reporting on, a huge-MAE config is pruned by a threshold pruner."""
    X, y, splits = toy_problem
    objective = _const_objective(1e6, X, y, splits, report_intermediate=True)
    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.ThresholdPruner(upper=1e3)
    )
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED


def test_no_report_means_no_pruning(toy_problem) -> None:
    """With reporting off, the same bad config runs to completion (not pruned)."""
    X, y, splits = toy_problem
    objective = _const_objective(1e6, X, y, splits, report_intermediate=False)
    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.ThresholdPruner(upper=1e3)
    )
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE

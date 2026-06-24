"""Registry of deployable models: short name -> spec (class, hardware kind, search space).

Single source of truth so the train-all orchestrator, the leaderboard, and the predict
tool agree on what "all models" means and how to build each one. ``kind`` drives the
ASAX queue + per-model tuning budget (neural -> GPU/fewer trials, tree -> CPU/more).

Baselines (global/group mean) are intentionally NOT here: they're sanity floors
evaluated for the leaderboard by the existing ``scripts/run_baselines.py``, not
deployable predictors, and they consume raw rows rather than the preprocessed array
every model here expects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import optuna

from ports_dfl.models.base import BaseModel
from ports_dfl.models.lgbm import LightGBMRegressorModel
from ports_dfl.models.linear import LinearRegressor
from ports_dfl.models.node import NODE
from ports_dfl.models.random_forest import RandomForestRegressorModel
from ports_dfl.models.realmlp import RealMLP
from ports_dfl.models.tabm import TabM
from ports_dfl.models.xgb import XGBoostRegressorModel
from ports_dfl.tuning.search_spaces import (
    suggest_lgbm,
    suggest_linear,
    suggest_node,
    suggest_realmlp,
    suggest_rf,
    suggest_tabm,
    suggest_xgb,
)

SuggestFn = Callable[[optuna.Trial], dict[str, Any]]


@dataclass(frozen=True)
class ModelSpec:
    """How to build and tune one model.

    Attributes:
        cls: the model class (a ``BaseModel`` subclass).
        kind: ``"neural"`` (GPU) or ``"tree"`` (CPU, fits with ``n_jobs=-1``).
        suggest_fn: the Optuna search-space function for this model.
        seed_kwarg: constructor kwarg name for the RNG seed — ``"seed"`` (linear/
            tabm/node) or ``"random_state"`` (trees/realmlp).
        early_stopping: whether ``fit`` consumes a validation set to stop early.
            ``False`` for RandomForest, so its final refit uses 100% of the data.
    """

    cls: type[BaseModel]
    kind: str
    suggest_fn: SuggestFn
    seed_kwarg: str
    early_stopping: bool = True


MODELS: dict[str, ModelSpec] = {
    "xgb": ModelSpec(XGBoostRegressorModel, "tree", suggest_xgb, "random_state"),
    "lgbm": ModelSpec(LightGBMRegressorModel, "tree", suggest_lgbm, "random_state"),
    "rf": ModelSpec(
        RandomForestRegressorModel, "tree", suggest_rf, "random_state", early_stopping=False
    ),
    "linear": ModelSpec(LinearRegressor, "neural", suggest_linear, "seed"),
    "realmlp": ModelSpec(RealMLP, "neural", suggest_realmlp, "random_state"),
    "tabm": ModelSpec(TabM, "neural", suggest_tabm, "seed"),
    "node": ModelSpec(NODE, "neural", suggest_node, "seed"),
}


def get_spec(name: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``name``, or raise with the known names.

    Raises:
        KeyError: if ``name`` is not a registered model.
    """
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(f"Unknown model {name!r}. Known: {sorted(MODELS)}") from None

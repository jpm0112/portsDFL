"""Optuna runner that wraps cross-validated objective evaluation.

The runner is parametrized over:
  - a ``model_factory(input_dim, **hp) -> BaseModel`` that builds a
    fresh model from a hyperparameter dict;
  - the suggestion function from ``search_spaces`` that produces those
    hyperparameters from a trial;
  - the dataset and CV splits.

Optimization minimizes mean validation MAE across folds.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from ports_dfl.data.encoders import build_preprocessor
from ports_dfl.metrics.regression import all_metrics

ModelFactory = Callable[..., Any]
SuggestFn = Callable[[optuna.Trial], dict[str, Any]]


def _evaluate_one_config(
    factory: ModelFactory,
    hp: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    categorical_strategy: str = "target",
    numeric_scaler: str = "standard",
    trial: optuna.Trial | None = None,
) -> dict[str, list[float]]:
    """Evaluate one hyperparameter configuration with K-fold CV.

    Args:
        trial: when provided, the running-mean val MAE is reported to the trial
            after each fold and the trial is pruned (``optuna.TrialPruned``) if
            the study's pruner says so. ``None`` (the default, used by every
            existing caller) disables reporting, preserving the original
            all-folds behavior exactly.

    Returns:
        Dict with one key per metric, each mapping to a list of per-fold values.
    """
    metric_lists: dict[str, list[float]] = {"mae": [], "rmse": [], "r2": [], "mape": []}
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_train = y.iloc[train_idx].to_numpy()
        y_val = y.iloc[val_idx].to_numpy()

        # build a FRESH preprocessor per fold: fitting only on this fold's train data
        # prevents validation info from leaking into the transform
        pre = build_preprocessor(
            categorical_strategy=categorical_strategy,
            numeric_scaler=numeric_scaler,
        )
        # passing y_train lets target-encoding learn from training labels only
        X_train = pre.fit_transform(X_train_raw, y_train).astype(np.float32)
        # transform (no fit) applies the already-learned encodings to val data
        X_val = pre.transform(X_val_raw).astype(np.float32)

        model = factory(input_dim=X_train.shape[1], **hp)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        metrics = all_metrics(y_val, preds)
        for k in metric_lists:
            metric_lists[k].append(metrics[k])

        # Report the partial CV score so the study's pruner can stop a clearly
        # losing config before all folds run. Only active when a trial is passed.
        if trial is not None:
            trial.report(float(np.mean(metric_lists["mae"])), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return metric_lists


def make_objective(
    factory: ModelFactory,
    suggest_fn: SuggestFn,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    extra_suggest: SuggestFn | None = None,
    report_intermediate: bool = False,
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective function for the given model + search space.

    Args:
        factory: callable that constructs the model from hyperparameters.
        suggest_fn: model-specific search space.
        X, y, splits: dataset and CV folds.
        extra_suggest: optional hook to add extra suggestions
            (e.g. categorical encoding strategy).
        report_intermediate: when True, report running-mean MAE after each fold
            so the study's pruner can stop weak trials early. Default False keeps
            the existing models' behavior unchanged (no pruning).

    Returns:
        An ``objective(trial) -> float`` minimizing mean val MAE.
    """

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_fn(trial)
        if extra_suggest is not None:
            hp.update(extra_suggest(trial))
        # pop these so they go to the preprocessor, not the model factory
        cat_strategy = hp.pop("categorical_strategy", "target")
        scaler = hp.pop("numeric_scaler", "standard")
        results = _evaluate_one_config(
            factory, hp, X, y, splits,
            categorical_strategy=cat_strategy, numeric_scaler=scaler,
            trial=trial if report_intermediate else None,
        )
        # stash per-fold metrics on the trial for later inspection / CSV dump
        trial.set_user_attr("mae_per_fold", results["mae"])
        trial.set_user_attr("rmse_per_fold", results["rmse"])
        trial.set_user_attr("r2_per_fold", results["r2"])
        trial.set_user_attr("mape_per_fold", results["mape"])
        return float(np.mean(results["mae"]))

    return objective


def run_study(
    study_name: str,
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    storage_dir: Path,
    seed: int = 42,
    pruner: optuna.pruners.BasePruner | None = None,
) -> optuna.Study:
    """Create or resume an Optuna study and run it for ``n_trials``.

    The study persists to a SQLite file under ``storage_dir`` so runs are
    resumable and tunable across sessions.

    Args:
        pruner: Optuna pruner to use; defaults to ``MedianPruner()`` (the prior
            behavior). Pass e.g. ``HyperbandPruner`` for the gradient-boosted
            benchmarks. A pruner only acts if the objective reports intermediate
            values (see ``make_objective(report_intermediate=...)``).
    """
    storage_dir.mkdir(parents=True, exist_ok=True)
    # as_posix() keeps the sqlite URL valid on Windows (forward slashes)
    storage_url = f"sqlite:///{(storage_dir / f'{study_name}.db').as_posix()}"
    # TPE Bayesian sampler; seeding makes runs reproducible
    sampler = optuna.samplers.TPESampler(seed=seed)
    # default: prune trials below median performance early
    if pruner is None:
        pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,  # resume an existing study instead of erroring
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def trials_to_dataframe(study: optuna.Study) -> pd.DataFrame:
    """Flatten a study's trials into a DataFrame for the results CSV."""
    return study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))

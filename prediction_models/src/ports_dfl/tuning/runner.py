"""Optuna runner that wraps cross-validated objective evaluation.

The runner is parametrized over:
  - a ``model_factory(input_dim, **hp) -> BaseModel`` that builds a
    fresh model from a hyperparameter dict;
  - the suggestion function from ``search_spaces`` that produces those
    hyperparameters from a trial;
  - the dataset and CV splits.

Optimization minimizes mean validation MAE across folds.
"""

from collections.abc import Callable  # `Callable` = the type of "a function you can call"
from pathlib import Path  # `Path` = an object-oriented way to work with file/folder paths
from typing import Any  # `Any` = "value of any type" (turns off type checking for that spot)

import numpy as np
import optuna
import pandas as pd

from ports_dfl.data.encoders import build_preprocessor
from ports_dfl.metrics.regression import all_metrics

# These two lines create type *aliases* (shorter names for long type hints).
# `Callable[..., Any]`: a function taking any arguments (`...`) and returning anything.
ModelFactory = Callable[..., Any]
# `Callable[[optuna.Trial], dict[str, Any]]`: a function that takes one `optuna.Trial`
# argument and returns a dict whose keys are str and whose values are anything.
SuggestFn = Callable[[optuna.Trial], dict[str, Any]]


# A leading underscore (`_evaluate_one_config`) is a Python convention meaning
# "private helper" — not part of this module's public API.
# The leading `_` does not enforce anything; it is just a hint to other coders.
def _evaluate_one_config(
    factory: ModelFactory,
    hp: dict[str, Any],
    X: pd.DataFrame,  # `X: pd.DataFrame` is a type hint: `X` is expected to be a pandas DataFrame
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],  # list of (train indices, val indices) pairs
    # Parameters below have defaults (`= "target"`), so callers may omit them.
    categorical_strategy: str = "target",
    numeric_scaler: str = "standard",
) -> dict[str, list[float]]:  # `-> ...` is the return type: a dict of metric-name -> list of floats
    """Evaluate one hyperparameter configuration with K-fold CV.

    Returns:
        Dict with one key per metric, each mapping to a list of per-fold values.
    """
    # Start with an empty list for each metric; we append one number per fold below.
    metric_lists: dict[str, list[float]] = {"mae": [], "rmse": [], "r2": [], "mape": []}
    # Unpack each (train_idx, val_idx) pair as we loop over the CV folds.
    for train_idx, val_idx in splits:
        # `.iloc[...]` selects rows by integer position (not by label) — used here
        # to grab the train/val rows for this fold.
        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        # `.to_numpy()` converts the pandas Series of targets into a plain numpy array.
        y_train = y.iloc[train_idx].to_numpy()
        y_val = y.iloc[val_idx].to_numpy()

        # Build a FRESH preprocessor inside the fold loop. This is important:
        # fitting it only on this fold's training data prevents validation info
        # from "leaking" into the transform (a classic data-leakage bug to avoid).
        pre = build_preprocessor(
            categorical_strategy=categorical_strategy,
            numeric_scaler=numeric_scaler,
        )
        # `fit_transform` learns encodings/scaling from train data AND applies them.
        # Passing `y_train` lets target-encoding learn from the training labels only.
        # `.astype(np.float32)` casts to 32-bit floats (smaller/faster for NN models).
        X_train = pre.fit_transform(X_train_raw, y_train).astype(np.float32)
        # `transform` (no fit) applies the ALREADY-learned encodings to val data.
        X_val = pre.transform(X_val_raw).astype(np.float32)

        # Build a new model from the hyperparameters. `**hp` "splats" the dict into
        # keyword arguments, e.g. {"lr": 0.01} becomes lr=0.01.
        # `X_train.shape[1]` is the number of columns (features) after preprocessing.
        model = factory(input_dim=X_train.shape[1], **hp)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        # Compute all metrics at once; `all_metrics` returns a dict keyed by metric name.
        metrics = all_metrics(y_val, preds)
        # Record this fold's value for every metric we track.
        for k in metric_lists:
            metric_lists[k].append(metrics[k])
    return metric_lists


def make_objective(
    factory: ModelFactory,
    suggest_fn: SuggestFn,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    extra_suggest: SuggestFn | None = None,
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective function for the given model + search space.

    Args:
        factory: callable that constructs the model from hyperparameters.
        suggest_fn: model-specific search space.
        X, y, splits: dataset and CV folds.
        extra_suggest: optional hook to add extra suggestions
            (e.g. categorical encoding strategy).

    Returns:
        An ``objective(trial) -> float`` minimizing mean val MAE.
    """

    # This is a "closure": `objective` is a function defined INSIDE another function.
    # It remembers (captures) the outer variables `factory`, `suggest_fn`, `X`, etc.,
    # so Optuna can later call `objective(trial)` with everything already wired in.
    def objective(trial: optuna.Trial) -> float:
        # Ask the search space for a hyperparameter dict for this specific trial.
        hp = suggest_fn(trial)
        # `is not None` is the idiomatic way to check an optional argument was provided.
        if extra_suggest is not None:
            # `dict.update(...)` merges another dict's keys into `hp` in place.
            hp.update(extra_suggest(trial))
        # `dict.pop(key, default)` removes and returns `key`'s value, or `default`
        # if absent. We pull these out so they go to the preprocessor, NOT the model
        # factory (which would not understand them).
        cat_strategy = hp.pop("categorical_strategy", "target")
        scaler = hp.pop("numeric_scaler", "standard")
        results = _evaluate_one_config(
            factory, hp, X, y, splits,
            categorical_strategy=cat_strategy, numeric_scaler=scaler,
        )
        # Persist per-trial details
        # `set_user_attr` stores extra info on the trial so it can be inspected later
        # (e.g. dumped to the results CSV) without affecting optimization.
        trial.set_user_attr("mae_per_fold", results["mae"])
        trial.set_user_attr("rmse_per_fold", results["rmse"])
        trial.set_user_attr("r2_per_fold", results["r2"])
        trial.set_user_attr("mape_per_fold", results["mape"])
        # Optuna minimizes this returned number. `np.mean` averages MAE across folds;
        # `float(...)` converts the numpy scalar to a plain Python float.
        return float(np.mean(results["mae"]))

    return objective  # return the function itself (not a call to it) for Optuna to use


def run_study(
    study_name: str,
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    storage_dir: Path,
    seed: int = 42,
) -> optuna.Study:
    """Create or resume an Optuna study and run it for ``n_trials``.

    The study persists to a SQLite file under ``storage_dir`` so runs are
    resumable and tunable across sessions.
    """
    # Create the output folder. `parents=True` makes any missing parent folders too;
    # `exist_ok=True` means "don't error if it already exists".
    storage_dir.mkdir(parents=True, exist_ok=True)
    # Build the SQLite connection URL. The `f"..."` is an f-string (lets you embed
    # `{expressions}` directly in text). `storage_dir / f'{study_name}.db'` uses
    # pathlib's `/` operator to join paths; `.as_posix()` gives forward-slash form
    # (important on Windows so the sqlite URL stays valid).
    storage_url = f"sqlite:///{(storage_dir / f'{study_name}.db').as_posix()}"
    # TPESampler = the smart Bayesian search strategy; seeding it makes runs reproducible.
    sampler = optuna.samplers.TPESampler(seed=seed)
    # MedianPruner stops unpromising trials early (below-median performance).
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",  # we are minimizing (lower MAE is better)
        load_if_exists=True,  # resume an existing study with this name instead of erroring
    )
    # Run the search: call `objective` `n_trials` times, letting the sampler choose hp.
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def trials_to_dataframe(study: optuna.Study) -> pd.DataFrame:
    """Flatten a study's trials into a DataFrame for the results CSV."""
    # `attrs=(...)` selects which trial fields become columns. Note this is a tuple
    # (the trailing comma-separated items in parentheses), not a list.
    return study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))

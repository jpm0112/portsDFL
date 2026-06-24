"""Optuna search spaces, one suggestion function per model.

Each function takes an ``optuna.Trial`` and returns a kwargs dict ready to
splat into the corresponding model's constructor.
"""

from typing import Any

import optuna


def suggest_linear(trial: optuna.Trial) -> dict[str, Any]:
    """Hyperparameter search space for the linear (Ridge) model."""
    return {
        # log scale spans the lr's orders of magnitude (1e-4 ... 1e-1)
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e0, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
    }


def suggest_realmlp(trial: optuna.Trial) -> dict[str, Any]:
    """Search space for RealMLP wrapper.

    Most defaults are excellent (the whole point of RealMLP) — we tune a
    small handful of parameters around them.
    """
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 384]),
        "depth": trial.suggest_int("depth", 2, 4),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.15, 0.2]),
        "lr": trial.suggest_float("lr", 5e-4, 5e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
    }


def suggest_tabm(trial: optuna.Trial) -> dict[str, Any]:
    """Search space for TabM (parameter-efficient ensemble of MLPs)."""
    return {
        # k_ensemble = how many MLP heads the TabM ensemble uses
        "k_ensemble": trial.suggest_categorical("k_ensemble", [8, 16, 32]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [192, 384, 768]),
        "depth": trial.suggest_int("depth", 2, 4),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.15]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
    }


def suggest_node(trial: optuna.Trial) -> dict[str, Any]:
    """Search space for NODE (neural oblivious decision ensembles).

    pytorch-tabular's NODE port currently only exposes ``sparsemoid`` as the
    bin function, so we tune the choice function while keeping bin fixed.
    """
    return {
        # step=2 restricts choices to 2, 4, 6
        "n_layers": trial.suggest_int("n_layers", 2, 6, step=2),
        "n_trees": trial.suggest_categorical("n_trees", [128, 256, 512]),
        "tree_depth": trial.suggest_int("tree_depth", 6, 8),
        # tune the choice function (sample routing) while the bin function stays fixed
        "choice_function": trial.suggest_categorical(
            "choice_function", ["entmax15", "sparsemax"]
        ),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }


def suggest_xgb(trial: optuna.Trial) -> dict[str, Any]:
    """Search space for the XGBoost benchmark.

    ``n_estimators`` is intentionally absent: it's fixed high in the model and
    early stopping selects the effective count, so tuning it would be redundant.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-1, 1e2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1e2, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1e2, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 1e1, log=True),
    }


def suggest_lgbm(trial: optuna.Trial) -> dict[str, Any]:
    """Search space for the LightGBM benchmark.

    Like XGBoost, ``n_estimators`` is fixed high + early stopping, so it isn't
    tuned here.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1e2, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1e2, log=True),
    }


def suggest_rf(trial: optuna.Trial) -> dict[str, Any]:
    """Search space for the RandomForest benchmark.

    Deliberately small: RandomForest is robust to tuning — more trees only help
    with diminishing returns, and the needle-movers are ``max_features`` and
    leaf size — so a few categorical choices suffice.
    """
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [300, 500, 800]),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    }

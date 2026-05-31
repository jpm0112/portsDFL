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

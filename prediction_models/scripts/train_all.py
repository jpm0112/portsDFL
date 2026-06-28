"""Train, tune, and persist every model as a portable artifact (Predict-then-Optimize).

For each requested model: run an Optuna study (trees: 40 trials + Hyperband pruning;
neural: 20 trials), re-evaluate the best config with full CV bookkeeping plus
out-of-fold predictions, then refit on all data and save the model alongside the
shared fitted preprocessor — so it reloads for inference without retraining.

Designed to be invoked once per model by an ASAX PBS job
(``python scripts/train_all.py --models xgb``); ``--models a,b,c`` loops locally.

Outputs per model:
    results/<model>/{cv_summary_tuned.csv, trials.csv, best_config.json, oof_predictions.csv}
    artifacts/{<model>.pkl, preprocessor.pkl, <model>.meta.json}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split

# Make the package importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ports_dfl.config import (  # noqa: E402
    ALL_FEATURES,
    HIGH_CARDINALITY_CATEGORICAL,
    LOW_CARDINALITY_CATEGORICAL,
    N_FOLDS,
    OPTUNA_DB_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    SEED,
    TARGET_COL,
)
from ports_dfl.data.encoders import build_preprocessor  # noqa: E402
from ports_dfl.data.loader import load_training_dataset, split_features_target  # noqa: E402
from ports_dfl.data.splits import make_cv_splits  # noqa: E402
from ports_dfl.metrics.regression import all_metrics, summarize_folds  # noqa: E402
from ports_dfl.models.base import BaseModel  # noqa: E402
from ports_dfl.models.registry import MODELS, ModelSpec, get_spec  # noqa: E402
from ports_dfl.tuning.runner import make_objective, run_study, trials_to_dataframe  # noqa: E402

logger = logging.getLogger("train_all")

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CAT_STRATEGY = "target"
NUMERIC_SCALER = "standard"
HOLDOUT_FRAC = 0.1  # internal val fraction for early-stopping models' final refit
SMOKE_TRIALS = 2
_VERSIONED_LIBS = ("xgboost", "lightgbm", "scikit-learn", "torch", "optuna", "category-encoders")


def _build_model(spec: ModelSpec, input_dim: int, seed: int, hp: dict) -> BaseModel:
    """Instantiate ``spec.cls`` with the model's own seed kwarg + tuned hyperparameters."""
    # spec.cls is type[BaseModel]; its abstract __init__ is argless, but every concrete
    # subclass accepts input_dim + its seed kwarg (asserted in tests/test_registry.py).
    return spec.cls(input_dim=input_dim, **{spec.seed_kwarg: seed, **hp})  # type: ignore[call-arg]


def _tune(
    spec: ModelSpec,
    name: str,
    study_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list,
    n_trials: int,
    out_dir: Path,
) -> dict:
    """Run the Optuna study; write trials.csv + best_config.json; return best params."""
    is_tree = spec.kind == "tree"

    def factory(input_dim: int, **hp: object) -> BaseModel:
        return _build_model(spec, input_dim, SEED, hp)

    objective = make_objective(
        factory=factory,
        suggest_fn=spec.suggest_fn,
        X=X,
        y=y,
        splits=splits,
        report_intermediate=is_tree,  # trees opt into fold-level Hyperband pruning
    )
    pruner = (
        optuna.pruners.HyperbandPruner(min_resource=1, max_resource=N_FOLDS, reduction_factor=3)
        if is_tree
        else None
    )
    study = run_study(
        study_name=study_name,
        objective=objective,
        n_trials=n_trials,
        storage_dir=OPTUNA_DB_DIR,
        seed=SEED,
        pruner=pruner,
    )
    trials_to_dataframe(study).to_csv(out_dir / "trials.csv", index=False)
    with open(out_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)
    logger.info("[%s] best trial #%d  MAE=%.3fh", name, study.best_trial.number, study.best_value)
    return dict(study.best_params)


def _evaluate_best(
    spec: ModelSpec,
    name: str,
    best_params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list,
    out_dir: Path,
) -> pd.DataFrame:
    """Re-CV the best config; write per-fold metrics + out-of-fold predictions."""
    fold_metrics: list[dict[str, float]] = []
    oof = np.full(len(y), np.nan)
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        pre = build_preprocessor(categorical_strategy=CAT_STRATEGY, numeric_scaler=NUMERIC_SCALER)
        y_train = y.iloc[train_idx].to_numpy()
        y_val = y.iloc[val_idx].to_numpy()
        X_train = pre.fit_transform(X.iloc[train_idx], y_train).astype(np.float32)
        X_val = pre.transform(X.iloc[val_idx]).astype(np.float32)

        # Distinct seed per fold so folds are independent draws, not identical reseeds.
        model = _build_model(spec, X_train.shape[1], SEED + fold_idx, best_params)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        oof[val_idx] = preds
        fold_metrics.append(all_metrics(y_val, preds))
        logger.info(
            "[%s] fold %d/%d  MAE=%.2fh", name, fold_idx + 1, len(splits), fold_metrics[-1]["mae"]
        )

    summary = summarize_folds(fold_metrics)
    summary.to_csv(out_dir / "cv_summary_tuned.csv")
    pd.DataFrame({"y_true": y.to_numpy(), "y_pred": oof}).to_csv(
        out_dir / "oof_predictions.csv", index=False
    )
    return summary


def _atomic_dump(obj: object, dest: Path, tag: str) -> None:
    """``joblib.dump`` to a per-tag temp file, then atomically replace ``dest``.

    Guarantees no reader ever sees a half-written ``dest``. This is safe for the
    shared ``preprocessor.pkl`` under parallel per-model PBS jobs *because* every job
    fits the preprocessor on the identical full dataset with identical settings, so
    whichever job wins the final rename is functionally interchangeable. (If the
    preprocessor ever becomes per-model, give one model sole ownership of writing it.)
    """
    tmp = dest.with_suffix(dest.suffix + f".{tag}.tmp")
    joblib.dump(obj, tmp)
    os.replace(tmp, dest)


def _save_vocab(X: pd.DataFrame, artifacts_dir: Path, tag: str) -> None:
    """Save the categorical vocabulary (values the encoders saw) so the predictor can warn
    when an input value is unseen. Atomic write, like the shared preprocessor."""
    cat_cols = [*LOW_CARDINALITY_CATEGORICAL, *HIGH_CARDINALITY_CATEGORICAL]
    vocab = {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in cat_cols}
    tmp = artifacts_dir / f".vocab.{tag}.tmp"
    tmp.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, artifacts_dir / "vocab.json")


def fit_full_and_save(
    spec: ModelSpec, name: str, best_params: dict, X: pd.DataFrame, y: pd.Series, artifacts_dir: Path
) -> Path:
    """Refit on all data and persist the model + shared preprocessor; return the artifact path."""
    pre = build_preprocessor(categorical_strategy=CAT_STRATEGY, numeric_scaler=NUMERIC_SCALER)
    X_all = pre.fit_transform(X, y).astype(np.float32)
    y_all = y.to_numpy()
    model = _build_model(spec, X_all.shape[1], SEED, best_params)
    if spec.early_stopping:
        # Carve a seeded holdout so early-stopping models can pick a stop point; without
        # a val set xgboost/lightgbm/neural would run to their max iterations and overfit.
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y_all, test_size=HOLDOUT_FRAC, random_state=SEED
        )
        model.fit(X_tr, y_tr, X_val, y_val)
    else:
        model.fit(X_all, y_all)  # RandomForest: no early stopping, train on 100%

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _atomic_dump(pre, artifacts_dir / "preprocessor.pkl", tag=name)
    _save_vocab(X, artifacts_dir, tag=name)
    artifact = artifacts_dir / f"{name}.pkl"
    model.save(artifact)
    return artifact


def _lib_versions() -> dict[str, str]:
    """Record installed versions of the libraries an artifact depends on (provenance)."""
    out: dict[str, str] = {}
    for pkg in _VERSIONED_LIBS:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out


def _write_meta(
    spec: ModelSpec,
    name: str,
    best_params: dict,
    summary: pd.DataFrame,
    artifact: Path,
    n_trials: int,
    artifacts_dir: Path,
) -> None:
    """Write the per-model manifest fragment (one file per model -> no parallel race)."""
    meta = {
        "name": name,
        "kind": spec.kind,
        "artifact": artifact.name,
        "preprocessor": "preprocessor.pkl",
        "n_trials": n_trials,
        "seed": SEED,
        "best_params": best_params,
        "cv": {
            metric: [float(summary.loc["mean", metric]), float(summary.loc["std", metric])]
            for metric in ("mae", "rmse", "r2", "mape")
        },
        "features": list(ALL_FEATURES),
        "target": TARGET_COL,
        "units": "hours",
        "lib_versions": _lib_versions(),
    }
    with open(artifacts_dir / f"{name}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _resolve_names(arg: str) -> list[str]:
    """Turn the --models argument into a validated list of registered model names."""
    names = list(MODELS) if arg == "all" else [n.strip() for n in arg.split(",") if n.strip()]
    for n in names:
        get_spec(n)  # fail loud on an unknown name
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + tune + persist models.")
    parser.add_argument("--models", default="all", help="comma-separated names or 'all'")
    parser.add_argument("--trials-tree", type=int, default=100)
    parser.add_argument("--trials-neural", type=int, default=40)
    parser.add_argument(
        "--smoke", action="store_true", help=f"tiny run ({SMOKE_TRIALS} trials) to test the pipeline"
    )
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    names = _resolve_names(args.models)

    logger.info("Loading dataset...")
    df = load_training_dataset()
    X, y = split_features_target(df)
    splits = make_cv_splits(df)
    logger.info("rows=%d folds=%d models=%s", len(df), len(splits), names)

    for name in names:
        spec = get_spec(name)
        n_trials = SMOKE_TRIALS if args.smoke else (
            args.trials_tree if spec.kind == "tree" else args.trials_neural
        )
        study_name = f"{name}_smoke" if args.smoke else name  # keep smoke out of the real study
        out_dir = RESULTS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=== %s (%s, %d trials) ===", name, spec.kind, n_trials)
        best = _tune(spec, name, study_name, X, y, splits, n_trials, out_dir)
        summary = _evaluate_best(spec, name, best, X, y, splits, out_dir)
        artifact = fit_full_and_save(spec, name, best, X, y, args.artifacts_dir)
        _write_meta(spec, name, best, summary, artifact, n_trials, args.artifacts_dir)
        logger.info(
            "[%s] saved -> %s  (CV MAE %.3f h)", name, artifact, summary.loc["mean", "mae"]
        )

    logger.info("Done. Artifacts in %s ; run scripts/compare.py for the leaderboard.", args.artifacts_dir)


if __name__ == "__main__":
    main()

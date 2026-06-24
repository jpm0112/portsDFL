"""Reload trained model artifacts and predict service times from a vessel CSV.

Loads the shared fitted preprocessor + each model's saved weights (written by
``scripts/train_all.py``) and applies them to new vessel rows. Pure Predict-then-
Optimize: no DFL / optimizer involved. The artifacts are portable, so this runs on
any machine with the package installed — no retraining.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ports_dfl.config import ALL_FEATURES
from ports_dfl.models.base import BaseModel
from ports_dfl.models.registry import get_spec

PREPROCESSOR_FILE = "preprocessor.pkl"


def _discover_metas(artifacts_dir: Path, models: list[str] | None) -> list[dict]:
    """Read the per-model manifest fragments, optionally filtered to ``models``.

    Raises:
        FileNotFoundError: if no fragments exist, or a requested model is absent.
    """
    metas = {
        meta["name"]: meta
        for meta_path in sorted(artifacts_dir.glob("*.meta.json"))
        for meta in [json.loads(meta_path.read_text(encoding="utf-8"))]
    }
    if not metas:
        raise FileNotFoundError(f"No model artifacts (*.meta.json) found in {artifacts_dir}.")
    if models is None:
        return list(metas.values())
    missing = [m for m in models if m not in metas]
    if missing:
        raise FileNotFoundError(f"No artifact for {missing}. Available: {sorted(metas)}")
    return [metas[m] for m in models]


def load_bundle(
    artifacts_dir: Path | str, models: list[str] | None = None
) -> tuple[object, dict[str, BaseModel], list[dict]]:
    """Load the fitted preprocessor + the requested trained models from ``artifacts_dir``.

    Args:
        artifacts_dir: directory written by ``scripts/train_all.py``.
        models: subset of model names to load; ``None`` loads every saved model.

    Returns:
        ``(preprocessor, {name: fitted model}, [manifest fragment, ...])``.

    Raises:
        FileNotFoundError: if the preprocessor or a requested model is missing.
    """
    artifacts_dir = Path(artifacts_dir)
    pre_path = artifacts_dir / PREPROCESSOR_FILE
    if not pre_path.exists():
        raise FileNotFoundError(f"Missing {PREPROCESSOR_FILE} in {artifacts_dir}.")
    preprocessor = joblib.load(pre_path)
    metas = _discover_metas(artifacts_dir, models)
    # Reconstruct an empty model of the right class, then restore its saved weights.
    # cls is type[BaseModel] (argless abstract __init__); concrete subclasses take input_dim.
    loaded: dict[str, BaseModel] = {}
    for meta in metas:
        cls = get_spec(meta["name"]).cls
        loaded[meta["name"]] = cls(input_dim=1).load(  # type: ignore[call-arg]
            artifacts_dir / meta["artifact"]
        )
    return preprocessor, loaded, metas


def predict_csv(
    input_csv: Path | str, artifacts_dir: Path | str, models: list[str] | None = None
) -> pd.DataFrame:
    """Predict service time (hours) for every row of ``input_csv``, one column per model.

    Args:
        input_csv: CSV containing at least the ``config.ALL_FEATURES`` columns.
        artifacts_dir: directory of trained artifacts.
        models: subset of models to run; ``None`` runs all saved models.

    Returns:
        DataFrame aligned to the input rows with one column per model plus an
        ``ensemble_mean`` column. Predictions are clamped at 0 (service time can't
        be negative).

    Raises:
        ValueError: if the input CSV is missing any required feature column.
    """
    df = pd.read_csv(input_csv)
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required feature columns: {missing}")

    preprocessor, loaded, _ = load_bundle(artifacts_dir, models)
    X = preprocessor.transform(df[ALL_FEATURES]).astype(np.float32)

    out = pd.DataFrame(index=df.index)
    for name, model in loaded.items():
        out[name] = np.clip(model.predict(X), a_min=0.0, a_max=None)
    out["ensemble_mean"] = out.mean(axis=1)
    return out

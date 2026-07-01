"""Portable vessel service-time predictor — tree models only (rf, xgb, lgbm).

Self-contained: loads the fitted preprocessor + tree estimators from ./artifacts and
predicts, with NO PyTorch and NO ports_dfl package (so it installs light and runs on
any machine). Feed a CSV of raw vessel fields (see sample_vessels.csv); it writes each
model's predicted berth service time in hours plus an ensemble mean.

    python predict.py                                   # runs on the bundled sample
    python predict.py --input my_vessels.csv --output preds.csv
    python predict.py --input my_vessels.csv --models rf,xgb
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from columns import ALL_FEATURES
from features import engineer, unseen_values

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
TREE_MODELS = ["rf", "xgb", "lgbm"]  # the models shipped in this bundle
BEST_MODEL = "rf"                    # leaderboard winner (lowest validation MAE)


def load_estimator(name: str):
    """Load the raw fitted estimator that train_all.py saved (a dict with 'estimator')."""
    pkl = ARTIFACTS / f"{name}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"Missing model artifact: {pkl}")
    return joblib.load(pkl)["estimator"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Raw vessel CSV -> tree-model service-time predictions (hours)."
    )
    parser.add_argument("--input", type=Path, default=HERE / "vessels.csv")
    parser.add_argument("--output", type=Path, default=HERE / "predictions.csv")
    parser.add_argument(
        "--models", default=None, help=f"comma-separated subset of {TREE_MODELS} (default: all)"
    )
    args = parser.parse_args()

    requested = [m.strip() for m in args.models.split(",")] if args.models else list(TREE_MODELS)
    unknown = [m for m in requested if m not in TREE_MODELS]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}. Available: {TREE_MODELS}")

    raw = pd.read_csv(args.input)
    features = engineer(raw)

    # Warn about categorical values not seen in training (they fall back to a default).
    vocab_path = ARTIFACTS / "vocab.json"
    if vocab_path.exists():
        for col, vals in unseen_values(features, json.loads(vocab_path.read_text("utf-8"))).items():
            print(f"  ! {col}: {vals} not seen in training; using that field's overall average")

    preprocessor = joblib.load(ARTIFACTS / "preprocessor.pkl")
    X = preprocessor.transform(features[ALL_FEATURES]).astype(np.float32)

    preds = pd.DataFrame(index=features.index)
    for name in requested:
        preds[name] = np.clip(np.asarray(load_estimator(name).predict(X)).ravel(), 0.0, None)
    preds["ensemble_mean"] = preds.mean(axis=1)

    # Show the vessel rows next to their predictions so the output is self-explanatory.
    result = pd.concat([raw.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    if "vessel_id" not in result.columns:  # auto-number inputs that lack an id
        result.insert(0, "vessel_id", range(1, len(result) + 1))
    result.to_csv(args.output, index=False)

    print(f"Wrote {len(result)} predictions (hours) to {args.output}\n")
    display = preds.round(2)
    display.index.name = "vessels"
    if BEST_MODEL in display.columns:
        display = display.rename(columns={BEST_MODEL: f"{BEST_MODEL} (BEST)"})
    print(display.to_string())

    if BEST_MODEL in preds.columns:
        if os.name == "nt":
            os.system("")  # enable ANSI colour codes in Windows terminals
        green, reset = "\033[1;92m", "\033[0m"
        print(
            f"\n{green}Best model = {BEST_MODEL} (lowest validation error). "
            f"Its predictions (hours): {preds[BEST_MODEL].round(2).tolist()}{reset}"
        )


if __name__ == "__main__":
    main()

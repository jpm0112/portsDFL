"""Predict vessel service times from a CSV of RAW vessel data.

THE entry point for the predictor tool. Feed it a CSV of raw vessel fields (see
README.md and sample_vessels.csv); it auto-engineers the model features and writes
each model's predicted berth service time (hours) plus an ensemble mean, reusing the
trained artifacts in ../artifacts. No retraining.

    python predict.py                                  # runs on the bundled sample
    python predict.py --input my_vessels.csv --output preds.csv
    python predict.py --input my_vessels.csv --models xgb,lgbm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# ponytail: current leaderboard winner (results/comparison.csv, sorted by MAE).
# Update if a retrain changes the ranking.
BEST_MODEL = "rf"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # prediction_models/src
sys.path.insert(0, str(HERE))  # this folder, so `features` imports

from features import engineer, unseen_values  # noqa: E402

from ports_dfl.inference import predict_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw vessel CSV -> per-model service-time predictions.")
    parser.add_argument(
        "--input", type=Path, default=HERE / "vessels.csv",
        help="CSV of raw vessel fields (default: vessels.csv in this folder)",
    )
    parser.add_argument("--output", type=Path, default=HERE / "predictions.csv")
    parser.add_argument("--artifacts", type=Path, default=HERE.parent / "artifacts")
    parser.add_argument("--models", default=None, help="comma-separated subset (default: all saved)")
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    features = engineer(raw)

    # Warn about categorical values not seen in training (they fall back to a default).
    vocab_path = args.artifacts / "vocab.json"
    if vocab_path.exists():
        for col, vals in unseen_values(features, json.loads(vocab_path.read_text("utf-8"))).items():
            print(f"  ! {col}: {vals} not seen in training; using that field's overall average")

    models = [m.strip() for m in args.models.split(",")] if args.models else None
    preds = predict_frame(features, args.artifacts, models)

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

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
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # prediction_models/src
sys.path.insert(0, str(HERE))  # this folder, so `features` imports

from features import engineer  # noqa: E402

from ports_dfl.inference import predict_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw vessel CSV -> per-model service-time predictions.")
    parser.add_argument(
        "--input", type=Path, default=HERE / "sample_vessels.csv",
        help="CSV of raw vessel fields (default: the bundled sample_vessels.csv)",
    )
    parser.add_argument("--output", type=Path, default=HERE / "predictions.csv")
    parser.add_argument("--artifacts", type=Path, default=HERE.parent / "artifacts")
    parser.add_argument("--models", default=None, help="comma-separated subset (default: all saved)")
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    features = engineer(raw)
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    preds = predict_frame(features, args.artifacts, models)

    # Show the vessel rows next to their predictions so the output is self-explanatory.
    result = pd.concat([raw.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} predictions (hours) to {args.output}\n")
    print(preds.round(2).to_string())


if __name__ == "__main__":
    main()

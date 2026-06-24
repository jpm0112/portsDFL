"""CLI: RAW vessel CSV -> auto-engineered features -> per-model service-time predictions.

Best-effort convenience wrapper around features.engineer + ports_dfl.inference. It
auto-derives the 17 model features from raw vessel fields (see README.md and
sample_raw_vessels.csv) then predicts. Some features are approximate (covid_era,
Calado diff) and rare categoricals fall back to the model prior -- for faithful
inputs, engineer the features yourself and use predict.py instead.

    python predict_from_raw.py                                   # runs on the bundled raw sample
    python predict_from_raw.py --input my_raw_vessels.csv --output preds.csv
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
    parser = argparse.ArgumentParser(description="RAW vessel CSV -> service-time predictions.")
    parser.add_argument(
        "--input", type=Path, default=HERE / "sample_raw_vessels.csv",
        help="CSV of RAW vessel fields (default: the bundled sample_raw_vessels.csv)",
    )
    parser.add_argument("--output", type=Path, default=HERE / "predictions.csv")
    parser.add_argument("--artifacts", type=Path, default=HERE.parent / "artifacts")
    parser.add_argument("--models", default=None, help="comma-separated subset (default: all saved)")
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    features = engineer(raw)
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    preds = predict_frame(features, args.artifacts, models)

    result = pd.concat([raw.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    result.to_csv(args.output, index=False)
    print(f"Auto-engineered {len(features.columns)} features; wrote {len(result)} predictions "
          f"(hours) to {args.output}\n")
    print(preds.round(2).to_string())


if __name__ == "__main__":
    main()

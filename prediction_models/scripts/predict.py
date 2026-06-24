"""CLI: predict vessel service times from a CSV using the trained model artifacts.

Reads a CSV of vessel rows (must contain the feature columns; a target column, if
present, is ignored), runs every saved model, and writes a CSV with one prediction
column per model plus an ``ensemble_mean``.

Usage:
    python scripts/predict.py --input vessels.csv --output predictions.csv
    python scripts/predict.py --input vessels.csv --models xgb,lgbm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ports_dfl.config import PROJECT_ROOT  # noqa: E402
from ports_dfl.inference import predict_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict service times for a vessel CSV.")
    parser.add_argument("--input", type=Path, required=True, help="CSV of vessel rows")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument(
        "--artifacts", type=Path, default=PROJECT_ROOT / "artifacts", help="trained-artifacts dir"
    )
    parser.add_argument(
        "--models", default=None, help="comma-separated subset (default: every saved model)"
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else None
    preds = predict_csv(args.input, args.artifacts, models)
    preds.to_csv(args.output, index=False)
    print(f"Wrote {len(preds)} predictions x {preds.shape[1]} columns to {args.output}")
    print(preds.head().to_string())


if __name__ == "__main__":
    main()

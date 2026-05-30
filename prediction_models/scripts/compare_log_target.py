"""Quick comparison of raw-target vs log-target across all four models.

Trains a *minimal* configuration of each model under both target schemes
on a single fold (no Optuna). Reports MAE, RMSE, R², MAPE side by side.

Usage:
    python scripts/compare_log_target.py
"""

# See compare.py for what `from __future__ import annotations` does: it lets us
# use modern type hints (like `list[dict]`) as plain text without runtime cost.
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np  # numpy: fast numeric arrays; `np` is the usual nickname
import pandas as pd

# Silence noisy UserWarnings (e.g. from sklearn/torch) so the printed table is clean.
warnings.filterwarnings("ignore", category=UserWarning)
# Put the project's src/ on the import path so `import ports_dfl...` works.
# (parents[1] = two folders up from this file; see compare.py for the breakdown.)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ports_dfl.config import RESULTS_DIR, SEED
from ports_dfl.data.encoders import build_preprocessor
from ports_dfl.data.loader import load_training_dataset, split_features_target
from ports_dfl.data.splits import make_cv_splits
from ports_dfl.metrics.regression import all_metrics
from ports_dfl.models.linear import LinearRegressor
from ports_dfl.models.log_target import LogTargetWrapper
from ports_dfl.models.node import NODE
from ports_dfl.models.realmlp import RealMLP
from ports_dfl.models.tabm import TabM


# A dict mapping a model name -> a "factory": a small function that, given the
# number of input features `d`, builds a fresh model instance. We use factories
# (lambdas) instead of pre-built objects so each run gets a brand-new model.
# 1e-2 is scientific notation for 0.01.
MODEL_FACTORIES = {
    "linear": lambda d: LinearRegressor(input_dim=d, max_epochs=200, lr=1e-2, weight_decay=1e-3),
    "realmlp": lambda d: RealMLP(input_dim=d, n_epochs=64),
    "tabm": lambda d: TabM(
        input_dim=d, k_ensemble=8, hidden_dim=192, depth=2, max_epochs=64, patience=12
    ),
    "node": lambda d: NODE(
        input_dim=d, n_layers=2, n_trees=128, tree_depth=6, max_epochs=64, patience=12
    ),
}


# Train one model on the training set and score it on the validation set.
def evaluate_one(model, X_train, y_train, X_val, y_val) -> dict:
    model.fit(X_train, y_train, X_val, y_val)  # learn from training data
    # Predict on validation inputs, then compute all metrics vs the true values.
    return all_metrics(y_val, model.predict(X_val))


def main() -> None:
    print("Loading dataset and preparing fold 0...")
    df = load_training_dataset()
    X, y = split_features_target(df)  # split table into inputs X and target y
    splits = make_cv_splits(df)       # list of (train_indices, val_indices) folds
    train_idx, val_idx = splits[0]    # use only the first fold (fold 0) for this quick test

    # Build the feature preprocessor. "target" encoding turns categorical columns
    # into numbers using the target. IMPORTANT: fit it ONLY on training data...
    pre = build_preprocessor(categorical_strategy="target")
    # .iloc[idx] selects rows by integer position. fit_transform learns the
    # transform from train rows AND applies it; .astype(np.float32) shrinks the
    # numbers to 32-bit floats (less memory, what the neural nets expect).
    Xt = pre.fit_transform(X.iloc[train_idx], y.iloc[train_idx]).astype(np.float32)
    # ...then only TRANSFORM the validation rows (no fitting -> no data leakage).
    Xv = pre.transform(X.iloc[val_idx]).astype(np.float32)
    yt = y.iloc[train_idx].to_numpy()  # pandas Series -> plain numpy array
    yv = y.iloc[val_idx].to_numpy()
    n_features = Xt.shape[1]  # .shape is (n_rows, n_cols); [1] = number of columns
    print(f"  features={n_features}, train={len(yt)}, val={len(yv)}\n")

    # `rows: list[dict]` is a type-hinted empty list we'll fill with result dicts.
    rows: list[dict] = []
    # .items() iterates (key, value) pairs of the dict: (model name, factory fn).
    for name, factory in MODEL_FACTORIES.items():
        print(f"Training {name} (raw-target)...")
        # factory(n_features) builds a fresh model sized to the feature count.
        raw = evaluate_one(factory(n_features), Xt, yt, Xv, yv)
        # `**raw` unpacks the metrics dict's keys/values into this new dict,
        # alongside the "model"/"target" labels we add manually.
        rows.append({"model": name, "target": "raw", **raw})

        print(f"Training {name} (log-target)...")
        # Same model, but wrapped so it trains/predicts on log1p(target). This is
        # the other half of the ablation: log-transformed target vs raw target.
        log_metrics = evaluate_one(
            LogTargetWrapper(factory(n_features)), Xt, yt, Xv, yv
        )
        rows.append({"model": name, "target": "log", **log_metrics})

    table = pd.DataFrame(rows)  # long-format table: one row per (model, target)
    # .pivot reshapes long -> wide: one row per model, columns split by target
    # ("raw"/"log") for each of the chosen metrics. Result has multi-level columns.
    pivot = table.pivot(index="model", columns="target", values=["mae", "mape", "r2"])
    # Flatten the 2-level column names (e.g. ("mae","raw")) into "mae_raw".
    pivot.columns = ["_".join(c) for c in pivot.columns]
    # Add convenience columns: how much log-target changed each metric vs raw.
    pivot["mae_delta"] = pivot["mae_log"] - pivot["mae_raw"]
    pivot["mape_delta_pct"] = (pivot["mape_log"] - pivot["mape_raw"]) * 100

    print("\n" + "=" * 80)
    print(" Raw vs log-target on fold 0 ")
    print("=" * 80)
    print(pivot.to_string(float_format=lambda v: f"{v:.3f}"))

    out = RESULTS_DIR / "log_target_comparison.csv"
    # Make sure the output folder exists. parents=True creates missing parent
    # folders too; exist_ok=True means "don't error if it already exists".
    out.parent.mkdir(parents=True, exist_ok=True)
    # Note: no index=False here, so the "model" row labels ARE written to the CSV.
    pivot.to_csv(out)
    print(f"\nWritten to {out}")


# Standard launcher: run main() only when this file is executed directly.
if __name__ == "__main__":
    main()

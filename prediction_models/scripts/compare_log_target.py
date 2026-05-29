"""Quick comparison of raw-target vs log-target across all four models.

Trains a *minimal* configuration of each model under both target schemes
on a single fold (no Optuna). Reports MAE, RMSE, R², MAPE side by side.

Usage:
    python scripts/compare_log_target.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
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


def evaluate_one(model, X_train, y_train, X_val, y_val) -> dict:
    model.fit(X_train, y_train, X_val, y_val)
    return all_metrics(y_val, model.predict(X_val))


def main() -> None:
    print("Loading dataset and preparing fold 0...")
    df = load_training_dataset()
    X, y = split_features_target(df)
    splits = make_cv_splits(df)
    train_idx, val_idx = splits[0]
    pre = build_preprocessor(categorical_strategy="target")
    Xt = pre.fit_transform(X.iloc[train_idx], y.iloc[train_idx]).astype(np.float32)
    Xv = pre.transform(X.iloc[val_idx]).astype(np.float32)
    yt = y.iloc[train_idx].to_numpy()
    yv = y.iloc[val_idx].to_numpy()
    n_features = Xt.shape[1]
    print(f"  features={n_features}, train={len(yt)}, val={len(yv)}\n")

    rows: list[dict] = []
    for name, factory in MODEL_FACTORIES.items():
        print(f"Training {name} (raw-target)...")
        raw = evaluate_one(factory(n_features), Xt, yt, Xv, yv)
        rows.append({"model": name, "target": "raw", **raw})

        print(f"Training {name} (log-target)...")
        log_metrics = evaluate_one(
            LogTargetWrapper(factory(n_features)), Xt, yt, Xv, yv
        )
        rows.append({"model": name, "target": "log", **log_metrics})

    table = pd.DataFrame(rows)
    pivot = table.pivot(index="model", columns="target", values=["mae", "mape", "r2"])
    pivot.columns = ["_".join(c) for c in pivot.columns]
    pivot["mae_delta"] = pivot["mae_log"] - pivot["mae_raw"]
    pivot["mape_delta_pct"] = (pivot["mape_log"] - pivot["mape_raw"]) * 100

    print("\n" + "=" * 80)
    print(" Raw vs log-target on fold 0 ")
    print("=" * 80)
    print(pivot.to_string(float_format=lambda v: f"{v:.3f}"))

    out = RESULTS_DIR / "log_target_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(out)
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()

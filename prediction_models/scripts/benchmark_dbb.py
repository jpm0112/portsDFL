"""Benchmark DBB DFL training time across DBAP sizes.

Times one round of:
  - per-solve cost (median of 10 warm-started solves)
  - one DFL epoch (30 train instances, batch 4, 1 epoch)

Quiet output: all PyEPO/blackbox stdout is suppressed.

Usage:
    python scripts/benchmark_dbb.py
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Hide all Python warning messages so the benchmark table stays clean.
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# The optimizers (bap_optim) now live in the sibling top-level package optimizers/src.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "optimizers" / "src"))

from ports_dfl.config import DEVICE, SEED
from ports_dfl.data.encoders import build_preprocessor
from ports_dfl.data.loader import load_training_dataset, split_features_target
from ports_dfl.data.splits import make_cv_splits
from bap_optim.discrete_bap import DiscreteBAP, generate_bap_instance
from ports_dfl.train.dfl_blackbox import DFLBlackboxConfig, train_dfl_blackbox


@contextlib.contextmanager
def silenced():
    """Suppress prints (including PyEPO's blackbox progress)."""
    buf_o, buf_e = io.StringIO(), io.StringIO()
    old_o, old_e = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buf_o, buf_e
    try:
        yield
    finally:
        # Always restore the originals, even if the body raised an error.
        sys.stdout, sys.stderr = old_o, old_e


class _Linear(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, x):
        return self.fc(x)


# Reshape a flat table of rows into groups of `n` rows, one group per BAP instance.
def reshape_instances(X, y, n):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    keep = (len(X) // n) * n  # drop leftover rows that don't fill a group
    return X[perm[:keep]].reshape(-1, n, X.shape[1]), y[perm[:keep]].reshape(-1, n)


def time_per_solve(model, n_solves=12):
    # Separate fixed-seed RNG so timing inputs are reproducible regardless of the
    # data shuffle above.
    rng = np.random.RandomState(0)
    times = []
    for k in range(n_solves):
        c = rng.uniform(10, 60, size=model.instance.n_vessels).astype(np.float32)
        t0 = time.time()
        model.setObj(c)
        model.solve()  # warm-started after the first call
        times.append(time.time() - t0)
    # Skip the first (cold) call: it includes one-time setup overhead. Return
    # median and mean of the rest, in milliseconds.
    return np.median(times[1:]) * 1000, np.mean(times[1:]) * 1000


def main() -> None:
    df = load_training_dataset()
    X, y = split_features_target(df)
    splits = make_cv_splits(df)
    train_idx, val_idx = splits[0]
    pre = build_preprocessor(categorical_strategy="target")
    y_train = y.iloc[train_idx].to_numpy()
    # Fit the preprocessing on train only; reuse it on val via transform.
    X_train = pre.fit_transform(X.iloc[train_idx], y_train).astype(np.float32)
    X_val = pre.transform(X.iloc[val_idx]).astype(np.float32)
    y_val = y.iloc[val_idx].to_numpy()
    d = X_train.shape[1]

    print(
        f"{'N':>3} {'M':>3} {'per_solve_med_ms':>17} {'per_solve_mean_ms':>18} "
        f"{'5_epoch_30_inst_s':>18}"
    )
    print("-" * 67)

    for N, M in [(5, 2), (6, 2), (8, 2), (8, 3), (10, 3)]:
        inst = generate_bap_instance(
            n_vessels=N, n_berths=M, horizon_hours=120.0, seed=SEED
        )
        optm = DiscreteBAP(inst)
        med, mean = time_per_solve(optm, n_solves=12)

        Xtr, ytr = reshape_instances(X_train, y_train, N)
        Xvl, yvl = reshape_instances(X_val, y_val, N)
        # Cap instance counts to bound runtime.
        Xtr, ytr = Xtr[:30], ytr[:30]
        Xvl, yvl = Xvl[:10], yvl[:10]

        head = _Linear(d).to(DEVICE)
        cfg = DFLBlackboxConfig(
            lr=1e-3, batch_size=4, max_epochs=5, patience=10, blackbox_lambd=1.0
        )
        t0 = time.time()
        with silenced():
            train_dfl_blackbox(head, Xtr, ytr, Xvl, yvl, inst, cfg)
        dfl_time = time.time() - t0

        print(
            f"{N:>3} {M:>3} {med:>15.1f}    {mean:>15.1f}    {dfl_time:>15.1f}"
        )


if __name__ == "__main__":
    main()

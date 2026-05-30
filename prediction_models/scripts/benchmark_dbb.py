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
# Add the project's "src" folder to Python's import search path so the
# `ports_dfl` package can be imported even though this script lives in a
# different folder. `Path(__file__)` is this file; `.resolve()` makes it an
# absolute path; `.parents[1]` goes up two levels (scripts/ -> prediction_models/).
# `sys.path.insert(0, ...)` puts our folder first so it is searched before others.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# The optimizers (bap_optim) now live in the sibling top-level package optimizers/src.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "optimizers" / "src"))

from ports_dfl.config import DEVICE, SEED
from ports_dfl.data.encoders import build_preprocessor
from ports_dfl.data.loader import load_training_dataset, split_features_target
from ports_dfl.data.splits import make_cv_splits
from bap_optim.discrete_bap import DiscreteBAP, generate_bap_instance
from ports_dfl.train.dfl_blackbox import DFLBlackboxConfig, train_dfl_blackbox


# The `@contextlib.contextmanager` decorator turns a function that `yield`s into
# something usable in a `with` block. Code before `yield` runs on entry, code
# after `yield` (here in `finally`) runs on exit. A decorator (the `@name` line
# above a function) wraps the function to give it extra behaviour.
@contextlib.contextmanager
def silenced():
    """Suppress prints (including PyEPO's blackbox progress)."""
    # io.StringIO() is an in-memory text buffer; anything "printed" into it is
    # captured instead of shown on screen.
    buf_o, buf_e = io.StringIO(), io.StringIO()
    # Remember the real stdout/stderr so we can put them back afterwards.
    old_o, old_e = sys.stdout, sys.stderr
    # Redirect prints (stdout) and errors (stderr) into the throwaway buffers.
    sys.stdout, sys.stderr = buf_o, buf_e
    try:
        yield  # run the body of the `with` block here
    finally:
        # Always restore the originals, even if the body raised an error.
        sys.stdout, sys.stderr = old_o, old_e


# A tiny PyTorch model: subclassing nn.Module is how you define a neural net.
class _Linear(nn.Module):
    def __init__(self, d):  # d = number of input features
        super().__init__()  # required: initialise the parent nn.Module
        self.fc = nn.Linear(d, 1)  # one linear layer mapping d inputs -> 1 output

    # forward() defines what the model computes when called on input x.
    def forward(self, x):
        return self.fc(x)


# Reshape a flat table of rows into groups of `n` rows, one group per BAP instance.
def reshape_instances(X, y, n):
    # default_rng(SEED) makes a reproducible random number generator.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))  # a shuffled ordering of all row indices
    # Drop leftover rows so the total divides evenly into groups of n.
    # (len(X) // n) is integer division; multiplying back gives the largest
    # multiple of n that fits.
    keep = (len(X) // n) * n
    # Take the first `keep` shuffled rows, then reshape into (num_instances, n, features)
    # for X and (num_instances, n) for the targets y. -1 means "infer this size".
    return X[perm[:keep]].reshape(-1, n, X.shape[1]), y[perm[:keep]].reshape(-1, n)


# Measure how long a single optimisation solve takes for the given model.
def time_per_solve(model, n_solves=12):
    # RandomState(0) is an older fixed-seed RNG (kept separate so timing inputs
    # are reproducible regardless of the data shuffle above).
    rng = np.random.RandomState(0)
    times = []  # collect the duration of each solve
    for k in range(n_solves):
        # Random objective costs, one per vessel, as 32-bit floats.
        c = rng.uniform(10, 60, size=model.instance.n_vessels).astype(np.float32)
        t0 = time.time()  # start the stopwatch
        model.setObj(c)  # set the objective for this solve
        model.solve()  # run the optimiser (warm-started after the first call)
        times.append(time.time() - t0)  # elapsed seconds for this solve
    # Skip the first (cold) call since it includes one-time setup overhead.
    # Return median and mean of the remaining solves, converted to milliseconds.
    return np.median(times[1:]) * 1000, np.mean(times[1:]) * 1000


# `-> None` is a type hint meaning this function returns nothing useful.
def main() -> None:
    # Load the dataset (a pandas DataFrame) and split it into features X and target y.
    df = load_training_dataset()
    X, y = split_features_target(df)
    # Build cross-validation splits and use the first one (index 0).
    # Each split is a (train_indices, val_indices) pair, unpacked below.
    splits = make_cv_splits(df)
    train_idx, val_idx = splits[0]
    pre = build_preprocessor(categorical_strategy="target")
    # .iloc[...] selects rows by position; .to_numpy() converts to a NumPy array.
    y_train = y.iloc[train_idx].to_numpy()
    # fit_transform learns the preprocessing on training data then applies it;
    # transform (below) reuses that same learned preprocessing on validation data.
    X_train = pre.fit_transform(X.iloc[train_idx], y_train).astype(np.float32)
    X_val = pre.transform(X.iloc[val_idx]).astype(np.float32)
    y_val = y.iloc[val_idx].to_numpy()
    d = X_train.shape[1]  # number of feature columns (model input size)

    # Print the table header. f-strings ({...} inside f"...") insert values;
    # `:>17` right-aligns the text in a field 17 characters wide so columns line up.
    print(
        f"{'N':>3} {'M':>3} {'per_solve_med_ms':>17} {'per_solve_mean_ms':>18} "
        f"{'5_epoch_30_inst_s':>18}"
    )
    print("-" * 67)  # a horizontal rule: "-" repeated 67 times

    # Sweep over several problem sizes: N vessels, M berths.
    for N, M in [(5, 2), (6, 2), (8, 2), (8, 3), (10, 3)]:
        # Build one optimisation problem instance of this size.
        inst = generate_bap_instance(
            n_vessels=N, n_berths=M, horizon_hours=120.0, seed=SEED
        )
        optm = DiscreteBAP(inst)  # wrap the instance in the solver object
        med, mean = time_per_solve(optm, n_solves=12)  # benchmark a single solve

        # Group the rows into instances of N vessels each (see reshape_instances).
        Xtr, ytr = reshape_instances(X_train, y_train, N)
        Xvl, yvl = reshape_instances(X_val, y_val, N)
        # Keep only the first 30 training and 10 validation instances to bound runtime.
        Xtr, ytr = Xtr[:30], ytr[:30]
        Xvl, yvl = Xvl[:10], yvl[:10]

        head = _Linear(d).to(DEVICE)  # fresh model moved to CPU/GPU as configured
        # Config for the DFL black-box training run (learning rate, batch size, etc.).
        cfg = DFLBlackboxConfig(
            lr=1e-3, batch_size=4, max_epochs=5, patience=10, blackbox_lambd=1.0
        )
        t0 = time.time()
        # Run training with all solver/PyEPO chatter suppressed (see silenced()).
        with silenced():
            train_dfl_blackbox(head, Xtr, ytr, Xvl, yvl, inst, cfg)
        dfl_time = time.time() - t0  # total wall-clock seconds for the training run

        # Print one results row. `:>15.1f` = right-aligned, width 15, 1 decimal place.
        print(
            f"{N:>3} {M:>3} {med:>15.1f}    {mean:>15.1f}    {dfl_time:>15.1f}"
        )


# This block runs main() only when the file is executed directly
# (e.g. `python benchmark_dbb.py`), not when it is imported as a module.
if __name__ == "__main__":
    main()

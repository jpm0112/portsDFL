"""Shared pytest fixtures.

Loads the dataset once per test session and exposes preprocessed numpy
arrays plus the train/val split for the first CV fold. Heavy fixtures
are session-scoped so model-specific tests can reuse them cheaply.
"""

# This file is named "conftest.py" on purpose: pytest auto-discovers it and
# makes every fixture defined here available to ALL test files in this folder
# (and subfolders) WITHOUT importing it. A test "uses" a fixture by listing the
# fixture's name as one of its function parameters (see how `df`, `Xy`, etc.
# are passed into tests in the sibling test_*.py files).

import numpy as np
import pytest

from ports_dfl.config import SEED
from ports_dfl.data.encoders import build_preprocessor
from ports_dfl.data.loader import load_training_dataset, split_features_target
from ports_dfl.data.splits import make_cv_splits


# @pytest.fixture marks a function as a reusable piece of test setup. Its return
# value is what tests receive. scope="session" means it runs only ONCE for the
# whole test run and the result is cached/shared across every test that asks for
# it -- important here because loading the dataset is expensive.
@pytest.fixture(scope="session")
def df():
    """Loaded training DataFrame (5,589 rows × 18 columns)."""
    return load_training_dataset()


# Fixtures can depend on other fixtures by naming them as parameters. Here `Xy`
# requests `df`, so pytest builds `df` first and feeds it in. This is the same
# session-cached DataFrame, so we are not reloading anything.
@pytest.fixture(scope="session")
def Xy(df):
    """Feature DataFrame and target Series."""
    return split_features_target(df)  # returns a (features, target) tuple


# Build the 5 cross-validation folds once. Passing a fixed SEED keeps the random
# split identical every run, so test results are reproducible.
@pytest.fixture(scope="session")
def cv_splits(df):
    """5-fold StratifiedKFold split indices (by Sitio)."""
    return make_cv_splits(df, seed=SEED)


@pytest.fixture(scope="session")
def first_fold_arrays(Xy, cv_splits):
    """Preprocessed numpy arrays for the first CV fold.

    Returns:
        ``(X_train, y_train, X_val, y_val, n_features)``
    """
    X, y = Xy  # unpack the (features, target) tuple from the Xy fixture
    train_idx, val_idx = cv_splits[0]  # use only fold 0 (the first split)
    # .iloc selects rows by integer position using the fold's index arrays.
    X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
    # Targets are converted to plain numpy arrays (models expect numpy, not pandas).
    y_train, y_val = y.iloc[train_idx].to_numpy(), y.iloc[val_idx].to_numpy()
    pre = build_preprocessor(categorical_strategy="target")
    # IMPORTANT: fit_transform is called on TRAIN only, then transform on VAL.
    # Fitting the preprocessor (e.g. target encoding) on validation data would
    # leak information from val into train -- a classic data-leakage bug. Doing
    # them separately here keeps the test setup honest.
    X_train = pre.fit_transform(X_train_raw, y_train).astype(np.float32)
    X_val = pre.transform(X_val_raw).astype(np.float32)
    # .shape[1] is the number of columns = number of input features.
    return X_train, y_train, X_val, y_val, X_train.shape[1]


# No scope= argument means the DEFAULT "function" scope: this fixture is rebuilt
# fresh for each test that uses it. It still reuses the cached, session-scoped
# `first_fold_arrays`, so it is cheap.
@pytest.fixture
def tiny_arrays(first_fold_arrays):
    """Tiny 64-row slice for fast over-fit smoke tests."""
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    # Slicing [:64] returns the first 64 rows. A small subset makes "does the
    # model train at all?" smoke tests run fast. NOTE: numpy slices are VIEWS,
    # not copies -- this is safe only because tests read these arrays and must
    # not modify them in place; a test that mutated a slice would corrupt the
    # shared session arrays for every later test.
    return X_train[:64], y_train[:64], X_val[:32], y_val[:32], n_features

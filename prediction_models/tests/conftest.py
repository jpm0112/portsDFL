"""Shared pytest fixtures.

Loads the dataset once per test session and exposes preprocessed numpy
arrays plus the train/val split for the first CV fold. Heavy fixtures
are session-scoped so model-specific tests can reuse them cheaply.
"""

import numpy as np
import pytest

from ports_dfl.config import SEED
from ports_dfl.data.encoders import build_preprocessor
from ports_dfl.data.loader import load_training_dataset, split_features_target
from ports_dfl.data.splits import make_cv_splits


@pytest.fixture(scope="session")
def df():
    """Loaded training DataFrame (5,589 rows × 18 columns)."""
    return load_training_dataset()


@pytest.fixture(scope="session")
def Xy(df):
    """Feature DataFrame and target Series."""
    return split_features_target(df)


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
    X, y = Xy
    train_idx, val_idx = cv_splits[0]
    X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx].to_numpy(), y.iloc[val_idx].to_numpy()
    pre = build_preprocessor(categorical_strategy="target")
    X_train = pre.fit_transform(X_train_raw, y_train).astype(np.float32)
    X_val = pre.transform(X_val_raw).astype(np.float32)
    return X_train, y_train, X_val, y_val, X_train.shape[1]


@pytest.fixture
def tiny_arrays(first_fold_arrays):
    """Tiny 64-row slice for fast over-fit smoke tests."""
    X_train, y_train, X_val, y_val, n_features = first_fold_arrays
    return X_train[:64], y_train[:64], X_val[:32], y_val[:32], n_features

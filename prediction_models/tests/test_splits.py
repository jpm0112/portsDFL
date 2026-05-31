"""Tests for the CV split helpers."""

import numpy as np

from ports_dfl.config import CV_STRATIFY_COL, N_FOLDS, TARGET_COL
from ports_dfl.data.splits import make_cv_splits, make_target_binned_cv_splits


def test_make_cv_splits_returns_expected_count(df) -> None:
    splits = make_cv_splits(df)
    assert len(splits) == N_FOLDS


def test_train_val_indices_partition_dataset(df) -> None:
    """For each fold, train_idx ∪ val_idx == full index, with no overlap."""
    splits = make_cv_splits(df)
    for train_idx, val_idx in splits:
        # No row is used for both training and validation in the same fold.
        assert len(set(train_idx) & set(val_idx)) == 0
        # Together the two index sets must cover every row exactly once.
        assert sorted(np.concatenate([train_idx, val_idx])) == list(range(len(df)))


def test_every_stratum_appears_in_every_fold(df) -> None:
    """Every level of the stratify column is present in train and val of each fold."""
    splits = make_cv_splits(df)
    levels = set(df[CV_STRATIFY_COL].unique())
    for train_idx, val_idx in splits:
        train_levels = set(df.iloc[train_idx][CV_STRATIFY_COL].unique())
        val_levels = set(df.iloc[val_idx][CV_STRATIFY_COL].unique())
        # Stratification means no level is ever dropped from either side.
        assert train_levels == levels
        assert val_levels == levels


def test_target_binned_splits_works(df) -> None:
    """Quantile-binned alternative split runs and partitions cleanly."""
    splits = make_target_binned_cv_splits(df, target_col=TARGET_COL, n_splits=5, n_bins=5)
    assert len(splits) == 5
    for train_idx, val_idx in splits:
        assert len(set(train_idx) & set(val_idx)) == 0


def test_seed_reproducibility(df) -> None:
    """Same seed yields identical splits."""
    s1 = make_cv_splits(df, seed=123)
    s2 = make_cv_splits(df, seed=123)
    # strict=True raises if the lists differ in length instead of truncating.
    for (t1, v1), (t2, v2) in zip(s1, s2, strict=True):
        assert np.array_equal(t1, t2)
        assert np.array_equal(v1, v2)

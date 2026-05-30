"""Tests for the CV split helpers."""

import numpy as np

# Config constants: which column to stratify by, how many folds, and the target.
from ports_dfl.config import CV_STRATIFY_COL, N_FOLDS, TARGET_COL
# The two split builders under test (from the still-to-be-written splits module).
from ports_dfl.data.splits import make_cv_splits, make_target_binned_cv_splits


# pytest runs every `test_*` function automatically. `df` is a fixture (defined
# in conftest.py) that pytest injects because it is named as a parameter.
def test_make_cv_splits_returns_expected_count(df) -> None:
    splits = make_cv_splits(df)
    # `assert` fails the test if False. We should get exactly N_FOLDS folds back.
    assert len(splits) == N_FOLDS


def test_train_val_indices_partition_dataset(df) -> None:
    """For each fold, train_idx ∪ val_idx == full index, with no overlap."""
    splits = make_cv_splits(df)
    # Each fold is a (train_indices, val_indices) pair; unpack it in the loop.
    for train_idx, val_idx in splits:
        # `set(a) & set(b)` is the INTERSECTION; it must be empty, i.e. no row
        # is used for both training and validation in the same fold.
        assert len(set(train_idx) & set(val_idx)) == 0
        # Together the two index sets must cover EVERY row exactly once.
        # `np.concatenate` glues the index arrays; sorting both sides lets us
        # compare them as the full ordered index 0..len(df)-1.
        assert sorted(np.concatenate([train_idx, val_idx])) == list(range(len(df)))


def test_every_stratum_appears_in_every_fold(df) -> None:
    """Every level of the stratify column is present in train and val of each fold."""
    splits = make_cv_splits(df)
    # All distinct values ("levels") of the stratify column across the whole data.
    levels = set(df[CV_STRATIFY_COL].unique())
    for train_idx, val_idx in splits:
        # `df.iloc[idx]` selects rows by integer POSITION; `.unique()` lists the
        # distinct stratify levels present in that slice.
        train_levels = set(df.iloc[train_idx][CV_STRATIFY_COL].unique())
        val_levels = set(df.iloc[val_idx][CV_STRATIFY_COL].unique())
        # Stratification means no level is ever dropped: both the train and val
        # portions of every fold must contain ALL levels.
        assert train_levels == levels
        assert val_levels == levels


def test_target_binned_splits_works(df) -> None:
    """Quantile-binned alternative split runs and partitions cleanly."""
    # Keyword arguments (name=value) make the call self-documenting and order-free.
    splits = make_target_binned_cv_splits(df, target_col=TARGET_COL, n_splits=5, n_bins=5)
    # We asked for 5 splits, so we should get 5.
    assert len(splits) == 5
    for train_idx, val_idx in splits:
        # Same no-overlap guarantee as the stratified splits above.
        assert len(set(train_idx) & set(val_idx)) == 0


def test_seed_reproducibility(df) -> None:
    """Same seed yields identical splits."""
    # Building the splits twice with the SAME seed must give identical results.
    s1 = make_cv_splits(df, seed=123)
    s2 = make_cv_splits(df, seed=123)
    # `zip(...)` walks both split lists in lockstep; `strict=True` (Python 3.10+)
    # raises if the two lists differ in length instead of silently truncating.
    # The double unpacking `(t1, v1), (t2, v2)` splits each paired fold at once.
    for (t1, v1), (t2, v2) in zip(s1, s2, strict=True):
        # `np.array_equal` is True only if the arrays have the same shape AND
        # every element matches — so identical train and val indices each fold.
        assert np.array_equal(t1, t2)
        assert np.array_equal(v1, v2)

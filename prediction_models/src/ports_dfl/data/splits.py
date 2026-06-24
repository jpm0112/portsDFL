"""Cross-validation split helpers.

Stratified K-fold by site (``config.CV_STRATIFY_COL``) so every berth site appears in
every fold, plus a target-binned variant for stratifying on the (continuous) target's
distribution. Both return plain ``(train_idx, val_idx)`` numpy-index arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ports_dfl.config import CV_STRATIFY_COL, N_FOLDS, SEED, TARGET_COL


def make_cv_splits(
    df: pd.DataFrame,
    seed: int | None = SEED,
    n_splits: int = N_FOLDS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold split indices, stratified by ``config.CV_STRATIFY_COL``.

    Args:
        df: the loaded dataset (must contain the stratify column).
        seed: shuffle seed; defaults to ``config.SEED`` for reproducible folds.
        n_splits: number of folds.

    Returns:
        List of ``(train_idx, val_idx)`` integer-index arrays, one tuple per fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(df)), df[CV_STRATIFY_COL]))


def make_target_binned_cv_splits(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    n_splits: int = N_FOLDS,
    n_bins: int = 5,
    seed: int | None = SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold using quantile bins of a continuous target as the strata.

    Useful when stratifying on the target's distribution rather than a category.

    Args:
        df: the loaded dataset.
        target_col: continuous column to bin.
        n_splits: number of folds.
        n_bins: number of quantile bins (duplicate edges are collapsed).
        seed: shuffle seed; defaults to ``config.SEED``.

    Returns:
        List of ``(train_idx, val_idx)`` integer-index arrays, one tuple per fold.
    """
    bins = pd.qcut(df[target_col], q=n_bins, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(df)), bins))

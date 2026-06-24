"""Data layer: load the dataset, build the preprocessor, make CV splits."""

from ports_dfl.data.encoders import build_preprocessor
from ports_dfl.data.loader import (
    feature_role_summary,
    load_training_dataset,
    split_features_target,
)
from ports_dfl.data.splits import make_cv_splits, make_target_binned_cv_splits

__all__ = [
    "build_preprocessor",
    "feature_role_summary",
    "load_training_dataset",
    "make_cv_splits",
    "make_target_binned_cv_splits",
    "split_features_target",
]

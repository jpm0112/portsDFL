"""Naive baselines used as sanity floors.

These define the worst that a model is allowed to do. If a real model
underperforms even GroupMeanBaseline, the pipeline (encoders, splits,
scaling) is broken and the model itself is not at fault.
"""

# `Path` is a class for filesystem paths that works on Windows/Linux/Mac alike.
from pathlib import Path

import joblib  # lightweight library for saving/loading Python objects to disk
import numpy as np  # numerical arrays; conventionally imported as `np`
import pandas as pd  # tables (DataFrames); conventionally imported as `pd`

# `BaseModel` is the abstract parent class that defines the shared interface
# (fit/predict/save/load). The classes below "inherit" from it.
from ports_dfl.models.base import BaseModel


# `class X(BaseModel):` means X is a subclass of BaseModel and must provide the
# methods BaseModel marked as abstract.
class GlobalMeanBaseline(BaseModel):
    """Predict the training-fold mean for every row."""

    # `def __init__(self) -> None:` is the constructor, run when you create the
    # object. `self` is the instance being built. `-> None` = returns nothing.
    def __init__(self) -> None:
        # Trailing-underscore name (`mean_`) is the scikit-learn convention for
        # "this attribute is set during fit()". `float | None` is a type hint
        # meaning the value is either a float or None (not yet fitted).
        self.mean_: float | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "GlobalMeanBaseline":  # noqa: D401
        """Compute and store the training mean."""
        # `np.asarray` turns y_train (list/Series/array) into a numpy array
        # without copying if it already is one; `.mean()` averages all values.
        # `float(...)` stores a plain Python float rather than a numpy scalar.
        self.mean_ = float(np.asarray(y_train).mean())
        return self  # return self so callers can chain: model.fit(...).predict(...)

    def predict(self, X) -> np.ndarray:
        # Guard: predicting before fit() would use a meaningless mean.
        if self.mean_ is None:
            raise RuntimeError("GlobalMeanBaseline must be fit before predict.")
        # `np.full(n, value)` builds an array of length n where every entry is
        # `value`. So every row gets the same global mean prediction.
        return np.full(len(X), self.mean_, dtype=float)

    # `Path | str` type hint: accepts either a Path object or a plain string.
    def save(self, path: Path | str) -> None:
        # Dump a small dict holding the only learned state (the mean) to disk.
        joblib.dump({"mean_": self.mean_}, Path(path))

    def load(self, path: Path | str) -> "GlobalMeanBaseline":
        # Read the saved dict back and restore the mean onto this instance.
        state = joblib.load(Path(path))
        self.mean_ = state["mean_"]
        return self


class GroupMeanBaseline(BaseModel):
    """Predict the per-group mean of the training fold.

    Designed to be called with a *raw* DataFrame X (not preprocessed) so
    that the grouping column is still available. For unseen group levels
    falls back to the global training mean.
    """

    # `group_col: str` = the name of the column to group rows by (e.g. "port").
    def __init__(self, group_col: str) -> None:
        self.group_col = group_col
        # `dict[str, float]` type hint: maps a group label -> its mean target.
        # Starts empty `{}`; filled in during fit().
        self.group_means_: dict[str, float] = {}
        self.global_mean_: float | None = None  # fallback mean for unseen groups

    def fit(
        self,
        X_train: pd.DataFrame,
        # `np.ndarray | pd.Series` = accepts a numpy array OR a pandas Series.
        y_train: np.ndarray | pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
    ) -> "GroupMeanBaseline":
        """Compute per-group means from the training fold."""
        # Fail fast if the grouping column is missing. `{self.group_col!r}` in
        # the f-string uses `!r` to show the repr (quotes around the name).
        if self.group_col not in X_train.columns:
            raise ValueError(f"group_col {self.group_col!r} not in X_train.columns")
        # `assign` returns a COPY of X_train with a new temporary column
        # `_target` holding the labels, so we can group features with targets
        # without mutating the caller's DataFrame.
        df = X_train.assign(_target=np.asarray(y_train))
        # Group rows by the group column, take the mean of `_target` within each
        # group, then `.to_dict()` -> {group_label: mean}.
        self.group_means_ = df.groupby(self.group_col)["_target"].mean().to_dict()
        # Overall mean used when a group wasn't seen in training.
        self.global_mean_ = float(np.asarray(y_train).mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.global_mean_ is None:
            raise RuntimeError("GroupMeanBaseline must be fit before predict.")
        group_vals = X[self.group_col]  # the group label for each row to predict
        # List comprehension: for each group label g, look up its learned mean;
        # `dict.get(g, default)` returns the global mean when g is unseen
        # (avoids a KeyError on new group levels at prediction time).
        return np.array(
            [self.group_means_.get(g, self.global_mean_) for g in group_vals],
            dtype=float,
        )

    def save(self, path: Path | str) -> None:
        # Persist all three pieces of learned state so the model can be fully
        # rebuilt later (the column name, per-group means, and global fallback).
        joblib.dump(
            {
                "group_col": self.group_col,
                "group_means_": self.group_means_,
                "global_mean_": self.global_mean_,
            },
            Path(path),
        )

    def load(self, path: Path | str) -> "GroupMeanBaseline":
        # Restore each saved field back onto this instance.
        state = joblib.load(Path(path))
        self.group_col = state["group_col"]
        self.group_means_ = state["group_means_"]
        self.global_mean_ = state["global_mean_"]
        return self

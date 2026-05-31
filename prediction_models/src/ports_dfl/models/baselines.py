"""Naive baselines used as sanity floors.

These define the worst that a model is allowed to do. If a real model
underperforms even GroupMeanBaseline, the pipeline (encoders, splits,
scaling) is broken and the model itself is not at fault.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ports_dfl.models.base import BaseModel


class GlobalMeanBaseline(BaseModel):
    """Predict the training-fold mean for every row."""

    def __init__(self) -> None:
        # Trailing-underscore name follows the scikit-learn "set during fit()"
        # convention; None means not yet fitted.
        self.mean_: float | None = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "GlobalMeanBaseline":  # noqa: D401
        """Compute and store the training mean."""
        self.mean_ = float(np.asarray(y_train).mean())
        return self

    def predict(self, X) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("GlobalMeanBaseline must be fit before predict.")
        return np.full(len(X), self.mean_, dtype=float)

    def save(self, path: Path | str) -> None:
        joblib.dump({"mean_": self.mean_}, Path(path))

    def load(self, path: Path | str) -> "GlobalMeanBaseline":
        state = joblib.load(Path(path))
        self.mean_ = state["mean_"]
        return self


class GroupMeanBaseline(BaseModel):
    """Predict the per-group mean of the training fold.

    Designed to be called with a *raw* DataFrame X (not preprocessed) so
    that the grouping column is still available. For unseen group levels
    falls back to the global training mean.
    """

    def __init__(self, group_col: str) -> None:
        self.group_col = group_col
        self.group_means_: dict[str, float] = {}
        self.global_mean_: float | None = None  # fallback mean for unseen groups

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | pd.Series | None = None,
    ) -> "GroupMeanBaseline":
        """Compute per-group means from the training fold."""
        if self.group_col not in X_train.columns:
            raise ValueError(f"group_col {self.group_col!r} not in X_train.columns")
        # Temporary `_target` column lets us group without mutating the caller's
        # DataFrame.
        df = X_train.assign(_target=np.asarray(y_train))
        self.group_means_ = df.groupby(self.group_col)["_target"].mean().to_dict()
        # Overall mean used when a group wasn't seen in training.
        self.global_mean_ = float(np.asarray(y_train).mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.global_mean_ is None:
            raise RuntimeError("GroupMeanBaseline must be fit before predict.")
        group_vals = X[self.group_col]
        # Fall back to the global mean for group levels unseen during fit
        # (avoids a KeyError at prediction time).
        return np.array(
            [self.group_means_.get(g, self.global_mean_) for g in group_vals],
            dtype=float,
        )

    def save(self, path: Path | str) -> None:
        joblib.dump(
            {
                "group_col": self.group_col,
                "group_means_": self.group_means_,
                "global_mean_": self.global_mean_,
            },
            Path(path),
        )

    def load(self, path: Path | str) -> "GroupMeanBaseline":
        state = joblib.load(Path(path))
        self.group_col = state["group_col"]
        self.group_means_ = state["group_means_"]
        self.global_mean_ = state["global_mean_"]
        return self

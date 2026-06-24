"""RandomForest regressor benchmark for service-time prediction.

A bagged-tree baseline for Predict-then-Optimize comparison only (never a DFL
backbone). Wraps ``sklearn.ensemble.RandomForestRegressor`` under the project's
BaseModel API.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ports_dfl.config import SEED
from ports_dfl.models.base import SklearnLikeModel


class RandomForestRegressorModel(SklearnLikeModel):
    """RandomForest regressor (BaseModel-compatible).

    RandomForest has no boosting rounds and therefore no early stopping, so
    ``fit`` ignores the validation set (accepted only for interface symmetry).
    """

    def __init__(
        self,
        input_dim: int | None = None,
        n_estimators: int = 500,
        max_features: float | str = "sqrt",
        min_samples_leaf: int = 1,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        random_state: int = SEED,
    ) -> None:
        # input_dim accepted for API symmetry; the forest infers it from X.
        self.input_dim = input_dim
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self._estimator = None  # built fresh in fit()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "RandomForestRegressorModel":
        """Fit the forest. ``X_val``/``y_val`` are ignored (no early stopping)."""
        self._estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_jobs=-1,
            random_state=self.random_state,
        )
        self._estimator.fit(X_train, y_train)
        return self

    def _init_kwargs(self) -> dict:
        """Constructor kwargs for serialization (excludes the fitted estimator)."""
        return {
            "input_dim": self.input_dim,
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "min_samples_leaf": self.min_samples_leaf,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "random_state": self.random_state,
        }

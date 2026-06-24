"""XGBoost regressor benchmark for service-time prediction.

A classic gradient-boosted-tree baseline used purely for Predict-then-Optimize
comparison; it is never a DFL backbone (trees aren't differentiable). Wraps
``xgboost.XGBRegressor`` under the project's BaseModel API so it reuses the same
CV / tuning / metrics machinery as the neural models.
"""

import numpy as np

from ports_dfl.config import SEED
from ports_dfl.models.base import SklearnLikeModel


class XGBoostRegressorModel(SklearnLikeModel):
    """XGBoost gradient-boosted-tree regressor (BaseModel-compatible).

    ``n_estimators`` is set deliberately high; when a validation set is given,
    early stopping on val MAE truncates the effective number of rounds, so the
    tuner does not need to search ``n_estimators``.

    Example:
        >>> model = XGBoostRegressorModel(learning_rate=0.05, max_depth=6)
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> preds = model.predict(X_val)
    """

    def __init__(
        self,
        input_dim: int | None = None,
        n_estimators: int = 2000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        early_stopping_rounds: int = 50,
        random_state: int = SEED,
    ) -> None:
        # input_dim is accepted for API symmetry with the other models; XGBoost
        # infers the feature count from X at fit time.
        self.input_dim = input_dim
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self._estimator = None  # built fresh in fit()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "XGBoostRegressorModel":
        """Fit XGBoost, early-stopping on val MAE when a validation set is given.

        Without a validation set, early stopping is disabled and all
        ``n_estimators`` rounds are trained.
        """
        # Local import so the package stays importable without xgboost installed.
        from xgboost import XGBRegressor

        use_early_stopping = X_val is not None and y_val is not None
        params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "gamma": self.gamma,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": self.random_state,
            "eval_metric": "mae",
        }
        # early_stopping_rounds is only valid alongside an eval_set; xgboost >= 2.0
        # takes both in the constructor, not in .fit().
        if use_early_stopping:
            params["early_stopping_rounds"] = self.early_stopping_rounds
        self._estimator = XGBRegressor(**params)
        if use_early_stopping:
            self._estimator.fit(
                X_train, y_train, eval_set=[(X_val, y_val)], verbose=False
            )
        else:
            self._estimator.fit(X_train, y_train)
        return self

    def _init_kwargs(self) -> dict:
        """Constructor kwargs for serialization (excludes the fitted estimator)."""
        return {
            "input_dim": self.input_dim,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "gamma": self.gamma,
            "early_stopping_rounds": self.early_stopping_rounds,
            "random_state": self.random_state,
        }

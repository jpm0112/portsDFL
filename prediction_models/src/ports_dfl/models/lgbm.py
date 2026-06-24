"""LightGBM regressor benchmark for service-time prediction.

A classic gradient-boosted-tree baseline for Predict-then-Optimize comparison
only (never a DFL backbone). Wraps ``lightgbm.LGBMRegressor`` under the project's
BaseModel API.
"""

import numpy as np

from ports_dfl.config import SEED
from ports_dfl.models.base import SklearnLikeModel


class LightGBMRegressorModel(SklearnLikeModel):
    """LightGBM gradient-boosted-tree regressor (BaseModel-compatible).

    As with the XGBoost wrapper, ``n_estimators`` is high and early stopping on
    val MAE (LightGBM's ``l1`` metric) selects the effective round count, so the
    tuner does not search ``n_estimators``.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        n_estimators: int = 2000,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 0.0,
        reg_alpha: float = 0.0,
        early_stopping_rounds: int = 50,
        random_state: int = SEED,
    ) -> None:
        # input_dim accepted for API symmetry; LightGBM infers it from X.
        self.input_dim = input_dim
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self._estimator = None  # built fresh in fit()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LightGBMRegressorModel":
        """Fit LightGBM, early-stopping on val MAE when a validation set is given."""
        # Local import so the package stays importable without lightgbm installed.
        import lightgbm as lgb

        use_early_stopping = X_val is not None and y_val is not None
        self._estimator = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=1,  # subsample only takes effect when freq >= 1
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            n_jobs=-1,
            verbose=-1,
            random_state=self.random_state,
        )
        if use_early_stopping:
            self._estimator.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="l1",
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
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
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "early_stopping_rounds": self.early_stopping_rounds,
            "random_state": self.random_state,
        }

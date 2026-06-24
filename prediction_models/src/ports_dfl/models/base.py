"""Abstract base class for all prediction models.

All concrete models (linear, RealMLP, TabM, NODE) inherit from
:class:`BaseModel` so that training, tuning, and evaluation code can treat
them interchangeably.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np


class BaseModel(ABC):
    """Common interface every model must implement.

    Subclasses are responsible for:
      - constructing themselves from hyperparameters in ``__init__``
      - fitting on a single (X_train, y_train, X_val, y_val) split
      - producing predictions for arbitrary X
      - serializing/deserializing weights for cross-fold reuse and
        downstream DFL fine-tuning.

    The class is intentionally small. Anything project-wide
    (training loop, metrics, CV) lives outside the model.
    """

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "BaseModel":
        """Train the model on one fold.

        Args:
            X_train: 2D array of preprocessed features.
            y_train: 1D target array.
            X_val: Optional validation features for early stopping.
            y_val: Optional validation targets.

        Returns:
            ``self`` for chaining.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return a 1D array of predictions for X."""

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Persist all state needed to restore the fitted model."""

    @abstractmethod
    def load(self, path: Path | str) -> "BaseModel":
        """Restore a previously saved model in place. Returns ``self``."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class SklearnLikeModel(BaseModel):
    """Base for scikit-learn-style regressors (RandomForest, XGBoost, LightGBM).

    Centralizes the parts that are identical across these wrappers — prediction
    with a not-fitted guard and joblib (de)serialization — so each concrete
    subclass only implements ``__init__``, ``fit``, and ``_init_kwargs``. The
    wrapped estimator lives on ``self._estimator`` (``None`` until ``fit``).

    These tree models are Predict-then-Optimize benchmarks only; they are never
    used as a DFL backbone (trees aren't differentiable).
    """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return a 1D prediction array; raise if called before ``fit``."""
        if getattr(self, "_estimator", None) is None:
            raise RuntimeError(f"{type(self).__name__} must be fit before predict.")
        # ravel() flattens any (n, 1) estimator output to the 1D BaseModel contract.
        return np.asarray(self._estimator.predict(X)).ravel()

    @abstractmethod
    def _init_kwargs(self) -> dict:
        """Constructor kwargs needed to rebuild this wrapper.

        Excludes the fitted estimator, which :meth:`save` stores separately.
        """

    def save(self, path: Path | str) -> None:
        """Persist the fitted estimator plus constructor kwargs via joblib."""
        if getattr(self, "_estimator", None) is None:
            raise RuntimeError("Nothing to save: model not fit.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"estimator": self._estimator, "init_kwargs": self._init_kwargs()}, path
        )

    def load(self, path: Path | str) -> "SklearnLikeModel":
        """Restore the estimator and kwargs saved by :meth:`save`. Returns self."""
        state = joblib.load(Path(path))
        self._estimator = state["estimator"]
        for key, value in state["init_kwargs"].items():
            setattr(self, key, value)
        return self

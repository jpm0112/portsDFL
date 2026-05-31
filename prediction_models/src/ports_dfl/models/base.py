"""Abstract base class for all prediction models.

All concrete models (linear, RealMLP, TabM, NODE) inherit from
:class:`BaseModel` so that training, tuning, and evaluation code can treat
them interchangeably.
"""

from abc import ABC, abstractmethod
from pathlib import Path

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

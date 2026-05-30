"""Abstract base class for all prediction models.

All concrete models (linear, RealMLP, TabM, NODE) inherit from
:class:`BaseModel` so that training, tuning, and evaluation code can treat
them interchangeably.
"""

# `abc` = "Abstract Base Classes". `ABC` is a base class that lets us define a
# class that CANNOT be instantiated on its own; `abstractmethod` is a decorator
# (a marker placed above a method) that says "every subclass MUST provide this
# method". Together they let us define an interface/contract.
from abc import ABC, abstractmethod

# `Path` is an object-oriented way to handle file paths (instead of plain strings).
from pathlib import Path

import numpy as np


# Inheriting from `ABC` makes BaseModel abstract: you can't do `BaseModel()`
# directly. Only a subclass that fills in all the @abstractmethod methods below
# can be created. This guarantees every model speaks the same "language".
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

    # @abstractmethod marks this as a method with no body here — a placeholder.
    # Any subclass MUST define its own `fit`, or Python refuses to create it.
    @abstractmethod
    def fit(
        self,
        # `self` is the model object itself (passed automatically when you call
        # `model.fit(...)`). The text after each `:` is a TYPE HINT — documentation
        # for what kind of value is expected; Python does not enforce it.
        # `np.ndarray` = a NumPy array (the standard numeric array type).
        X_train: np.ndarray,
        y_train: np.ndarray,
        # `np.ndarray | None = None` means: either a NumPy array OR `None`
        # (nothing). The `= None` makes it OPTIONAL — callers can omit it.
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        # `-> "BaseModel"` is the RETURN type hint: this returns a model object.
        # It's in quotes ("forward reference") because the class is still being
        # defined on this line, so the name isn't fully available yet.
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
        # No body: subclasses supply the real training logic.

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return a 1D array of predictions for X."""

    # `path: Path | str` = accept either a Path object or a plain string.
    # `-> None` means this method returns nothing (it just writes a file).
    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Persist all state needed to restore the fitted model."""

    @abstractmethod
    def load(self, path: Path | str) -> "BaseModel":
        """Restore a previously saved model in place. Returns ``self``."""

    # `__repr__` is a special "dunder" (double-underscore) method Python calls to
    # get the text shown when you print/inspect the object (e.g. in a REPL).
    def __repr__(self) -> str:
        # f-string: text in quotes prefixed with `f`, where `{...}` is replaced by
        # the value inside. `self.__class__.__name__` is the subclass's name, so a
        # LinearModel instance prints as "LinearModel()".
        return f"{self.__class__.__name__}()"

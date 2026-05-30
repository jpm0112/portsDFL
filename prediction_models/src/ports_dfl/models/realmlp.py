"""RealMLP wrapper around pytabkit.RealMLP_TD_Regressor.

Tier 2 of the model lineup. RealMLP (Holzmüller et al., NeurIPS 2024) is a
plain MLP shipped with carefully calibrated default hyperparameters that
match or beat boosted trees on most tabular benchmarks. We wrap it to fit
the BaseModel interface so it can be slotted into the same training, CV,
and tuning machinery as the other models.

Reference: https://github.com/dholzmueller/pytabkit
"""

from pathlib import Path  # Path = object-oriented filesystem paths (cleaner than raw strings)

import joblib  # library for saving/loading Python objects to disk (used by scikit-learn too)
import numpy as np

from ports_dfl.config import DEVICE
from ports_dfl.models.base import BaseModel


# class RealMLP(BaseModel): "(BaseModel)" means RealMLP INHERITS from BaseModel.
# It must implement BaseModel's @abstractmethod methods (fit/predict/save/load)
# or Python will refuse to create an instance of it.
class RealMLP(BaseModel):
    """Wrap pytabkit's RealMLP regressor under the project's BaseModel API.

    pytabkit owns the training loop, optimizer, scheduler, and validation
    early stopping internally. This wrapper exposes a small set of useful
    overrides via ``__init__`` and routes everything else to RealMLP's
    well-tuned defaults.
    """

    # __init__ is the constructor: it runs when you write RealMLP(...).
    # "self" is the new object being built; we attach settings onto it below.
    # Type hints like "int | None = None" mean: an int OR None, defaulting to None.
    # "-> None" means this method returns nothing.
    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        depth: int | None = None,
        dropout: float | None = None,
        lr: float | None = None,
        weight_decay: float | None = None,
        n_epochs: int = 256,
        random_state: int = 42,
    ) -> None:
        # input_dim is accepted for API symmetry with PyTorch models; pytabkit
        # infers it from X at fit time.
        # Storing each argument on self so other methods (like _build) can read them.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.random_state = random_state
        # Leading underscore (_estimator) is a convention meaning "internal, don't touch
        # from outside". None for now; the real pytabkit model is created later in fit().
        self._estimator = None  # pytabkit estimator built lazily in fit()

    # Leading underscore = "private helper" by convention. No type hint on the return
    # because pytabkit may not be installed when this file is first read.
    def _build(self):
        """Construct the pytabkit estimator with TD defaults plus overrides."""
        # Local import keeps the package importable even before pytabkit is
        # installed (e.g. during early test discovery).
        # Importing INSIDE the function (not at the top) delays the dependency
        # until the moment we actually need it.
        from pytabkit import RealMLP_TD_Regressor

        # Pick GPU ("cuda") if available, otherwise CPU. This is a ternary expression:
        # value_if_true if condition else value_if_false.
        device = "cuda" if DEVICE.type == "cuda" else "cpu"
        # A dict of keyword arguments we'll hand to pytabkit. We start with the
        # always-present settings, then conditionally add overrides below.
        kwargs = {
            "n_epochs": self.n_epochs,
            "device": device,
            "random_state": self.random_state,
            "verbosity": 0,  # 0 = silent; suppress pytabkit's training chatter
        }
        # Only pass non-None overrides so we keep RealMLP's tuned defaults.
        # [value] * n builds a list with that value repeated n times. So a hidden_dim
        # of 256 with depth 3 -> [256, 256, 256], i.e. three hidden layers of width 256.
        # "self.depth or 3" uses 3 when depth is None or 0 (falsy).
        if self.hidden_dim is not None:
            kwargs["hidden_sizes"] = [self.hidden_dim] * (self.depth or 3)
        if self.dropout is not None:
            kwargs["p_drop"] = self.dropout  # pytabkit names its dropout arg "p_drop"
        if self.lr is not None:
            kwargs["lr"] = self.lr
        if self.weight_decay is not None:
            kwargs["wd"] = self.weight_decay  # pytabkit names weight decay "wd"
        # **kwargs unpacks the dict into named arguments: f(**{"a": 1}) == f(a=1).
        return RealMLP_TD_Regressor(**kwargs)

    # The "RealMLP" return type is in quotes because the class isn't fully defined
    # yet at the point Python reads this line (a "forward reference").
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "RealMLP":
        # Build a fresh estimator each fit so repeated fits don't carry old state.
        self._estimator = self._build()
        # If a validation set was provided, pass it so pytabkit can early-stop on it.
        if X_val is not None and y_val is not None:
            self._estimator.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            self._estimator.fit(X_train, y_train)
        # Return self so callers can chain, e.g. model.fit(...).predict(...).
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Guard: predicting before fit() would crash deeper in pytabkit with a
        # confusing error, so we fail early with a clear message.
        if self._estimator is None:
            raise RuntimeError("RealMLP must be fit before predict.")
        # np.asarray makes sure we have a numpy array; .ravel() flattens it to 1D
        # (e.g. shape (n, 1) -> (n,)) to match the BaseModel "1D predictions" contract.
        return np.asarray(self._estimator.predict(X)).ravel()

    def save(self, path: Path | str) -> None:
        if self._estimator is None:
            raise RuntimeError("Nothing to save: model not fit.")
        # Save BOTH the trained estimator and the constructor kwargs, so load()
        # can fully reconstruct this wrapper. Path(path) accepts a str or Path.
        joblib.dump(
            {"estimator": self._estimator, "init_kwargs": self._init_kwargs()},
            Path(path),
        )

    def load(self, path: Path | str) -> "RealMLP":
        # joblib.load returns the dict we saved above.
        state = joblib.load(Path(path))
        self._estimator = state["estimator"]
        # Restore each saved constructor setting onto self. .items() yields
        # (key, value) pairs; setattr(self, k, v) is the same as self.<k> = v
        # but with the attribute name held in a variable.
        for k, v in state["init_kwargs"].items():
            setattr(self, k, v)
        return self

    def _init_kwargs(self) -> dict:
        """Capture constructor kwargs for serialization."""
        # A plain dict mirroring __init__'s arguments. Note: this does NOT include
        # _estimator (the trained model), which save() stores separately.
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "random_state": self.random_state,
        }

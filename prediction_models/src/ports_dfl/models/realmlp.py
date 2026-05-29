"""RealMLP wrapper around pytabkit.RealMLP_TD_Regressor.

Tier 2 of the model lineup. RealMLP (Holzmüller et al., NeurIPS 2024) is a
plain MLP shipped with carefully calibrated default hyperparameters that
match or beat boosted trees on most tabular benchmarks. We wrap it to fit
the BaseModel interface so it can be slotted into the same training, CV,
and tuning machinery as the other models.

Reference: https://github.com/dholzmueller/pytabkit
"""

from pathlib import Path

import joblib
import numpy as np

from ports_dfl.config import DEVICE
from ports_dfl.models.base import BaseModel


class RealMLP(BaseModel):
    """Wrap pytabkit's RealMLP regressor under the project's BaseModel API.

    pytabkit owns the training loop, optimizer, scheduler, and validation
    early stopping internally. This wrapper exposes a small set of useful
    overrides via ``__init__`` and routes everything else to RealMLP's
    well-tuned defaults.
    """

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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.random_state = random_state
        self._estimator = None  # pytabkit estimator built lazily in fit()

    def _build(self):
        """Construct the pytabkit estimator with TD defaults plus overrides."""
        # Local import keeps the package importable even before pytabkit is
        # installed (e.g. during early test discovery).
        from pytabkit import RealMLP_TD_Regressor

        device = "cuda" if DEVICE.type == "cuda" else "cpu"
        kwargs = {
            "n_epochs": self.n_epochs,
            "device": device,
            "random_state": self.random_state,
            "verbosity": 0,
        }
        # Only pass non-None overrides so we keep RealMLP's tuned defaults.
        if self.hidden_dim is not None:
            kwargs["hidden_sizes"] = [self.hidden_dim] * (self.depth or 3)
        if self.dropout is not None:
            kwargs["p_drop"] = self.dropout
        if self.lr is not None:
            kwargs["lr"] = self.lr
        if self.weight_decay is not None:
            kwargs["wd"] = self.weight_decay
        return RealMLP_TD_Regressor(**kwargs)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "RealMLP":
        self._estimator = self._build()
        if X_val is not None and y_val is not None:
            self._estimator.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            self._estimator.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("RealMLP must be fit before predict.")
        return np.asarray(self._estimator.predict(X)).ravel()

    def save(self, path: Path | str) -> None:
        if self._estimator is None:
            raise RuntimeError("Nothing to save: model not fit.")
        joblib.dump(
            {"estimator": self._estimator, "init_kwargs": self._init_kwargs()},
            Path(path),
        )

    def load(self, path: Path | str) -> "RealMLP":
        state = joblib.load(Path(path))
        self._estimator = state["estimator"]
        for k, v in state["init_kwargs"].items():
            setattr(self, k, v)
        return self

    def _init_kwargs(self) -> dict:
        """Capture constructor kwargs for serialization."""
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

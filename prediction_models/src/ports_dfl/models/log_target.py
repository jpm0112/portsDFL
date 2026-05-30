"""Log-target wrapper for any BaseModel.

Service time is right-skewed (mean 40.8h, max 298h). Training on log(y)
typically improves both R² and MAPE substantially since:

    - the noise structure is closer to multiplicative than additive;
    - extreme tail values stop dominating the MSE gradient;
    - the back-transform exp(x̂) is naturally non-negative, removing a
      common source of nonsense predictions on short-stay vessels.

Caveat: predictions are exponentiated before metric computation, so the
metrics reported are still in target units (hours).
"""

from pathlib import Path

import joblib  # simple library for saving/loading Python objects to disk
import numpy as np

from ports_dfl.models.base import BaseModel


class LogTargetWrapper(BaseModel):
    """Wraps any BaseModel so it learns log(target) and predicts exp(out).

    Args:
        inner: Any BaseModel (linear, RealMLP, TabM, NODE).
        offset: small positive value added before log to handle y=0 edge cases
            (target should already be > 0 here, but kept for safety).

    Usage:
        >>> from ports_dfl.models.linear import LinearRegressor
        >>> base = LinearRegressor(input_dim=42)
        >>> model = LogTargetWrapper(base)
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> preds_in_hours = model.predict(X_val)   # back on the original scale
    """

    # `inner: BaseModel` = the wrapped model. This is the "decorator/wrapper"
    # pattern: LogTargetWrapper is itself a BaseModel, but delegates the real
    # work to `inner` while transforming y on the way in and out.
    def __init__(self, inner: BaseModel, offset: float = 1.0) -> None:
        self.inner = inner
        self.offset = float(offset)  # added before log so log(0) -> log(offset) is finite

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LogTargetWrapper":
        # Transform targets to log space BEFORE training. We compute the log in
        # float64 (double precision) for accuracy, then store as float32 because
        # that's what the torch models train with. np.asarray ensures it's a numpy
        # array (e.g. if a Python list or pandas Series was passed in).
        y_train_log = np.log(np.asarray(y_train, dtype=np.float64) + self.offset).astype(
            np.float32
        )
        # Only transform the validation targets if they were actually provided.
        if y_val is not None:
            y_val_log = np.log(np.asarray(y_val, dtype=np.float64) + self.offset).astype(
                np.float32
            )
        else:
            y_val_log = None
        # The inner model trains entirely in log space; X is passed through unchanged.
        self.inner.fit(X_train, y_train_log, X_val, y_val_log)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # The inner model outputs predictions in LOG space; undo the transform here.
        log_preds = self.inner.predict(X)
        # np.clip(a, lo, hi) forces every value into [lo, hi]. We bound to [-10, 10]
        # so np.exp() can't blow up to inf (exp(10) ~= 22026, a sane ceiling).
        # Then subtract offset to invert the `+ offset` we added before log in fit().
        return np.exp(np.clip(log_preds, -10.0, 10.0)) - self.offset

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)  # ensure the folder exists
        # Save inner using its own mechanism, plus our offset.
        # The inner model gets its own file alongside ours (".inner" appended), so
        # each model serializes itself however it needs (torch, sklearn, etc.).
        inner_path = path.with_suffix(path.suffix + ".inner")
        self.inner.save(inner_path)
        # joblib.dump writes a small metadata file: our offset plus the inner
        # model's class/module names (handy for debugging) and where it was saved.
        joblib.dump(
            {
                "offset": self.offset,
                "inner_class": type(self.inner).__name__,  # e.g. "LinearRegressor"
                "inner_module": type(self.inner).__module__,  # e.g. "ports_dfl.models.linear"
                "inner_path": str(inner_path),
            },
            path,
        )

    def load(self, path: Path | str) -> "LogTargetWrapper":
        meta = joblib.load(Path(path))
        self.offset = float(meta["offset"])
        # Restore inner via the path it wrote.
        # NOTE: this relies on self.inner already being an instance of the correct
        # model class (load() restores weights in place, it does not construct the
        # object). Callers must build the wrapper with a matching inner first.
        self.inner.load(meta["inner_path"])
        return self

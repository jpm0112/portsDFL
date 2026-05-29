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

import joblib
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

    def __init__(self, inner: BaseModel, offset: float = 1.0) -> None:
        self.inner = inner
        self.offset = float(offset)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LogTargetWrapper":
        y_train_log = np.log(np.asarray(y_train, dtype=np.float64) + self.offset).astype(
            np.float32
        )
        if y_val is not None:
            y_val_log = np.log(np.asarray(y_val, dtype=np.float64) + self.offset).astype(
                np.float32
            )
        else:
            y_val_log = None
        self.inner.fit(X_train, y_train_log, X_val, y_val_log)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_preds = self.inner.predict(X)
        # Clip to avoid overflow if the inner model produces extreme values
        return np.exp(np.clip(log_preds, -10.0, 10.0)) - self.offset

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save inner using its own mechanism, plus our offset.
        inner_path = path.with_suffix(path.suffix + ".inner")
        self.inner.save(inner_path)
        joblib.dump(
            {
                "offset": self.offset,
                "inner_class": type(self.inner).__name__,
                "inner_module": type(self.inner).__module__,
                "inner_path": str(inner_path),
            },
            path,
        )

    def load(self, path: Path | str) -> "LogTargetWrapper":
        meta = joblib.load(Path(path))
        self.offset = float(meta["offset"])
        # Restore inner via the path it wrote
        self.inner.load(meta["inner_path"])
        return self

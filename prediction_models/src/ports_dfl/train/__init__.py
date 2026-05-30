"""Training loops: predict-then-optimize (PtO) and decision-focused (DFL)."""

# This file is the package's __init__.py: importing `ports_dfl.train` runs this,
# and these re-exports let callers write `from ports_dfl.train import train_pto`
# instead of reaching into the submodules (pto.py / dfl_blackbox.py) directly.
from ports_dfl.train.dfl_blackbox import (
    DFLBlackboxConfig,
    DFLBlackboxResult,
    train_dfl_blackbox,
)
from ports_dfl.train.pto import TrainConfig, TrainResult, predict_pto, train_pto

# `__all__` lists the public names exported by `from ports_dfl.train import *`.
# It also documents the intended public API of this package.
__all__ = [
    "DFLBlackboxConfig",
    "DFLBlackboxResult",
    "TrainConfig",
    "TrainResult",
    "predict_pto",
    "train_dfl_blackbox",
    "train_pto",
]

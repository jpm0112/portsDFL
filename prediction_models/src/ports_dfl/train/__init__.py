"""Training loops: predict-then-optimize (PtO) and decision-focused (DFL)."""

from ports_dfl.train.pto import TrainConfig, TrainResult, predict_pto, train_pto

# DFL blackbox training needs the optimizer stack (pyepo + a MILP solver), which the
# slim prediction-only env deliberately omits. Import it when present; otherwise expose
# only PtO so the prediction models import without the solver deps. Re-raise on any
# *other* missing module so a genuine import bug still fails loudly.
try:
    from ports_dfl.train.dfl_blackbox import (
        DFLBlackboxConfig,
        DFLBlackboxResult,
        train_dfl_blackbox,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"pyepo", "pyomo", "gurobipy"}:
        raise
    DFLBlackboxConfig = DFLBlackboxResult = train_dfl_blackbox = None  # type: ignore[assignment,misc]

__all__ = [
    "DFLBlackboxConfig",
    "DFLBlackboxResult",
    "TrainConfig",
    "TrainResult",
    "predict_pto",
    "train_dfl_blackbox",
    "train_pto",
]

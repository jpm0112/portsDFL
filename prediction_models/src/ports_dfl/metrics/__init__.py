"""Regression and decision-quality metrics."""

# Re-export the metric functions at the package level so callers can write
# `from ports_dfl.metrics import mae` instead of the longer full path.
from ports_dfl.metrics.regression import mae, mape, r2, rmse, summarize_folds

# __all__ defines the public API: it controls what `from ports_dfl.metrics
# import *` brings in, and signals to readers/tools which names are intended
# for external use. (Note: `all_metrics` exists in regression.py but is not
# re-exported here — see REPORTED.)
__all__ = ["mae", "mape", "r2", "rmse", "summarize_folds"]

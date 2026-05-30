"""
Baseline predictive distributions for benchmarking M0.

Two baselines are implemented:

1. NoPoolingBaseline:
   For each (vessel, berth, service) cell observed in training, fit a
   Lognormal by maximum likelihood on the cell's log(svc) values. Cells
   with fewer than `min_n` training observations are not fit; their
   predictions fall back to the global Lognormal. This represents the
   "use only the local data, ignore the rest" stance.

2. FullPoolingBaseline:
   Single Lognormal fit to all training log(svc), used for every test row
   regardless of cell. This represents the "ignore all category structure"
   stance.

Both expose a `predictive_samples(test_df, n_draws)` method that returns
an array of shape (n_test, n_draws) of service_time_hours samples, so
the evaluator can use the same downstream code that consumes M0 samples.
"""

from __future__ import annotations  # lets us write type hints like `int | None` on older Pythons

from dataclasses import dataclass  # @dataclass auto-writes __init__/__repr__ for simple "record" classes

import numpy as np
import pandas as pd


# Cells with fewer training observations than this fall back to the
# global pool under the no-pooling baseline. The choice of 2 is the bare
# minimum for a sample standard deviation to be defined.
MIN_CELL_N_FOR_LOCAL_FIT = 2


# @dataclass is a "decorator": it modifies the class below it. Here it turns the two
# annotated fields (mu, sigma) into constructor arguments, so we can write
# _LognormalParams(mu=..., sigma=...) without writing an __init__ by hand.
# `mu: float` is a type hint: it documents that mu should be a float (not enforced at runtime).
@dataclass
class _LognormalParams:
    """Mean and standard deviation on the log scale (Lognormal sufficient stats)."""

    mu: float     # mean of log(service_time)
    sigma: float  # standard deviation of log(service_time)


def _fit_lognormal(log_y: np.ndarray, fallback_sigma: float) -> _LognormalParams:
    """
    Maximum-likelihood fit of a Lognormal to a vector of log-service times.

    Input:
        log_y: 1D array of log(service_time) values for the cell.
        fallback_sigma: sigma to use if log_y has < 2 elements.

    Output:
        _LognormalParams with mu = mean, sigma = sample sd (or fallback).

    Description:
        For 0- or 1-sized cells we use a fallback sigma (typically the
        global residual sd) so the predictive distribution is still
        well-defined. Single-sample cells use sample mean and fallback sd.
    """
    # Empty cell: no data to estimate a mean from, so mu is "not a number" (np.nan).
    if len(log_y) == 0:
        return _LognormalParams(mu=np.nan, sigma=fallback_sigma)
    mu = float(np.mean(log_y))  # MLE of the log-scale mean is just the sample average
    # With only one observation the sample standard deviation is undefined,
    # so borrow the fallback sigma (usually the global spread).
    if len(log_y) < 2:
        return _LognormalParams(mu=mu, sigma=fallback_sigma)
    # ddof=1 gives the unbiased "sample" standard deviation (divide by n-1, not n).
    sigma = float(np.std(log_y, ddof=1))
    return _LognormalParams(mu=mu, sigma=sigma)


class FullPoolingBaseline:
    """
    Predicts the same Lognormal for every test row.

    Attributes:
        params: global _LognormalParams fit to all training log(svc).
    """

    # __init__ runs when you create the object (e.g. FullPoolingBaseline(df)). `self` is
    # the instance being built; attributes set on `self` persist on the object.
    # `log_target_col: str = "log_service_time"` is a parameter with a default value.
    def __init__(self, train_df: pd.DataFrame, log_target_col: str = "log_service_time"):
        """
        Fit the global Lognormal at construction time.

        Input:
            train_df: training rows with log_target_col present.
        """
        log_y = train_df[log_target_col].to_numpy()  # pull the log-target column out as a plain numpy array
        # Fit one Lognormal to ALL training rows: its mean and sd ARE the model.
        self.params = _LognormalParams(mu=float(np.mean(log_y)), sigma=float(np.std(log_y, ddof=1)))

    def predictive_samples(self, test_df: pd.DataFrame, n_draws: int, rng: np.random.Generator) -> np.ndarray:
        """
        Draw n_draws Lognormal predictive samples for each test row.

        Input:
            test_df: any DataFrame; only its row count matters.
            n_draws: number of predictive draws per row.
            rng: numpy random generator for reproducibility.

        Output:
            ndarray of shape (n_test, n_draws), in service_time_hours units.

        Description:
            Same Lognormal for every row, so this is just exp(Normal samples).
        """
        n_test = len(test_df)
        # Draw on the log scale (Normal), then exponentiate to get hours.
        # A Lognormal variable is exactly exp(Normal), so this is a Lognormal sample.
        log_samples = rng.normal(self.params.mu, self.params.sigma, size=(n_test, n_draws))
        return np.exp(log_samples)  # back to service_time_hours units


class NoPoolingBaseline:
    """
    Per-cell Lognormal fit; cells with too few observations fall back to the global pool.

    Attributes:
        cell_params: dict (vessel_idx, berth_idx, service_idx) -> _LognormalParams.
        fallback: FullPoolingBaseline used for cells absent or too sparse.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        log_target_col: str = "log_service_time",
        min_n: int = MIN_CELL_N_FOR_LOCAL_FIT,
    ):
        """
        Fit one Lognormal per training cell with at least `min_n` observations.

        Input:
            train_df: training rows with vessel_idx/berth_idx/service_idx and
                      log_target_col columns.
            log_target_col: name of the log-target column.
            min_n: minimum cell size to fit a local Lognormal; below this,
                   the row falls back to the global pool at predict time.
        """
        # The global (full-pooling) fit doubles as the safety net for sparse cells.
        self.fallback = FullPoolingBaseline(train_df, log_target_col=log_target_col)
        # Annotated empty dict: keys are 3-tuples of category indices, values are fitted params.
        self.cell_params: dict[tuple[int, int, int], _LognormalParams] = {}
        # groupby(...) splits the training rows into one group per unique
        # (vessel, berth, service) combination; we look only at the log-target column.
        grouped = train_df.groupby(["vessel_idx", "berth_idx", "service_idx"])[log_target_col]
        # Each loop iteration gives `cell` (the 3-tuple key) and `log_y` (that group's values).
        for cell, log_y in grouped:
            arr = log_y.to_numpy()
            # Only fit a local Lognormal when the cell has enough data; otherwise we
            # leave it out of the dict so predict-time falls back to the global pool.
            if len(arr) >= min_n:
                self.cell_params[cell] = _fit_lognormal(arr, fallback_sigma=self.fallback.params.sigma)

    def predictive_samples(self, test_df: pd.DataFrame, n_draws: int, rng: np.random.Generator) -> np.ndarray:
        """
        Per-row predictive samples, using the local Lognormal if available.

        Input/Output: see FullPoolingBaseline.predictive_samples.

        Description:
            For each test row, look up the (vessel, berth, service) cell.
            If a local fit exists, draw from it; otherwise fall back to the
            global Lognormal. This mirrors how a naive analyst would
            backstop sparse cells without partial pooling.
        """
        out = np.empty((len(test_df), n_draws))  # pre-allocate the (n_test, n_draws) output array
        # zip(...) pairs up the three columns row-by-row into (vessel, berth, service) tuples,
        # matching the tuple keys we stored in self.cell_params.
        keys = list(zip(test_df["vessel_idx"].to_numpy(), test_df["berth_idx"].to_numpy(), test_df["service_idx"].to_numpy()))
        # enumerate gives both the row index `i` and the key, so we can write into out[i].
        for i, key in enumerate(keys):
            # dict.get returns None if the cell was never fit (too sparse or unseen)...
            params = self.cell_params.get(key)
            if params is None:
                params = self.fallback.params  # ...so fall back to the global Lognormal
            log_samples = rng.normal(params.mu, params.sigma, size=n_draws)  # draw on log scale
            out[i] = np.exp(log_samples)  # convert this row's draws to hours
        return out

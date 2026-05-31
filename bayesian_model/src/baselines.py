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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Cells with fewer training observations than this fall back to the global pool
# under the no-pooling baseline. 2 is the minimum for a sample SD to be defined.
MIN_CELL_N_FOR_LOCAL_FIT = 2


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
    # Empty cell: no data to estimate a mean from.
    if len(log_y) == 0:
        return _LognormalParams(mu=np.nan, sigma=fallback_sigma)
    mu = float(np.mean(log_y))
    # With only one observation the sample SD is undefined, so borrow the fallback.
    if len(log_y) < 2:
        return _LognormalParams(mu=mu, sigma=fallback_sigma)
    sigma = float(np.std(log_y, ddof=1))
    return _LognormalParams(mu=mu, sigma=sigma)


class FullPoolingBaseline:
    """
    Predicts the same Lognormal for every test row.

    Attributes:
        params: global _LognormalParams fit to all training log(svc).
    """

    def __init__(self, train_df: pd.DataFrame, log_target_col: str = "log_service_time"):
        """
        Fit the global Lognormal at construction time.

        Input:
            train_df: training rows with log_target_col present.
        """
        log_y = train_df[log_target_col].to_numpy()
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
        # A Lognormal variable is exactly exp(Normal): draw on the log scale, then exp.
        log_samples = rng.normal(self.params.mu, self.params.sigma, size=(n_test, n_draws))
        return np.exp(log_samples)


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
        self.cell_params: dict[tuple[int, int, int], _LognormalParams] = {}
        grouped = train_df.groupby(["vessel_idx", "berth_idx", "service_idx"])[log_target_col]
        for cell, log_y in grouped:
            arr = log_y.to_numpy()
            # Only fit locally when the cell has enough data; otherwise leave it out
            # so predict-time falls back to the global pool.
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
        out = np.empty((len(test_df), n_draws))
        # (vessel, berth, service) tuples matching the keys stored in self.cell_params.
        keys = list(zip(test_df["vessel_idx"].to_numpy(), test_df["berth_idx"].to_numpy(), test_df["service_idx"].to_numpy()))
        for i, key in enumerate(keys):
            # None if the cell was never fit (too sparse or unseen) -> global fallback.
            params = self.cell_params.get(key)
            if params is None:
                params = self.fallback.params
            log_samples = rng.normal(params.mu, params.sigma, size=n_draws)
            out[i] = np.exp(log_samples)
        return out

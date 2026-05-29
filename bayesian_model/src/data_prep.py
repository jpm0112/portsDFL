"""
Data preparation for the Bayesian Hierarchical Model of vessel service time.

Loads the BHM training CSV, integer-encodes the three hierarchical factors
(vessel type, berth/Sitio, service), and splits time-based into train/test.

All encodings are fit on the training rows only. Test rows that contain
unseen category levels are mapped to a reserved out-of-vocabulary index
(OOV); the model treats OOV as zero group offset (i.e., predicts at the
global mean alpha0). This keeps evaluation honest while still allowing
inference for cells that were absent in training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


# Sentinel index reserved for categories not seen during training.
OOV_INDEX = -1


@dataclass(frozen=True)
class Encoding:
    """
    Lookup tables produced by encode_categoricals.

    Each dict maps the original category string to a non-negative integer
    index used by the PyMC model. The OOV_INDEX sentinel is reserved for
    levels that appear only in test data.

    Attributes:
        vessel: vessel-type string -> int index
        berth: berth (Sitio) string -> int index
        service: service string -> int index
        n_vessel: number of distinct training vessel types
        n_berth: number of distinct training berths
        n_service: number of distinct training services
    """

    vessel: Dict[str, int]
    berth: Dict[str, int]
    service: Dict[str, int]
    n_vessel: int
    n_berth: int
    n_service: int


def load_bhm(path: str | Path) -> pd.DataFrame:
    """
    Load the BHM training dataset from CSV.

    Input:
        path: filesystem path to training_dataset_bhm.csv (typically the
              local copy at bayesian_model/training_dataset_bhm.csv).

    Output:
        pandas.DataFrame with all 16 source columns and no row filtering.

    Description:
        Thin wrapper around pandas.read_csv with UTF-8 encoding. Performs
        no preprocessing so that downstream steps (encoding, splitting) are
        explicit and testable in isolation.
    """
    return pd.read_csv(path, encoding="utf-8")


def time_split(
    df: pd.DataFrame,
    train_year_max: int = 2024,
    year_col: str = "atraque_year",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train and test by berthing year.

    Input:
        df: full dataset.
        train_year_max: last year (inclusive) included in the training set.
        year_col: column holding the berthing year.

    Output:
        (train_df, test_df) with disjoint, contiguous year ranges. Indices
        are reset on both partitions for clean integer positions.

    Description:
        Honest holdout for forecasting: rows with year <= train_year_max go
        to train, the rest to test. This matches the deployment scenario
        where the DFL model predicts service time for future arrivals using
        only past data.
    """
    train_mask = df[year_col] <= train_year_max
    train_df = df.loc[train_mask].reset_index(drop=True)
    test_df = df.loc[~train_mask].reset_index(drop=True)
    return train_df, test_df


def encode_categoricals(
    train_df: pd.DataFrame,
    vessel_col: str = "Tipo nave (agrupado)",
    berth_col: str = "Sitio",
    service_col: str = "Servicio",
) -> Encoding:
    """
    Build integer index lookups for the three hierarchical factors.

    Input:
        train_df: training rows only (encodings must not see test data).
        vessel_col / berth_col / service_col: column names for each factor.

    Output:
        Encoding dataclass with one dict per factor (string -> non-negative
        int) and the count of unique training levels.

    Description:
        Categories are sorted alphabetically before assigning indices for
        deterministic behavior across runs. The OOV sentinel is *not* part
        of these dicts; it is injected at apply time by apply_encoding for
        any category not in the lookup.
    """
    vessel_map = {v: i for i, v in enumerate(sorted(train_df[vessel_col].unique()))}
    berth_map = {v: i for i, v in enumerate(sorted(train_df[berth_col].unique()))}
    service_map = {v: i for i, v in enumerate(sorted(train_df[service_col].unique()))}
    return Encoding(
        vessel=vessel_map,
        berth=berth_map,
        service=service_map,
        n_vessel=len(vessel_map),
        n_berth=len(berth_map),
        n_service=len(service_map),
    )


def apply_encoding(
    df: pd.DataFrame,
    encoding: Encoding,
    vessel_col: str = "Tipo nave (agrupado)",
    berth_col: str = "Sitio",
    service_col: str = "Servicio",
) -> pd.DataFrame:
    """
    Add integer index columns to a DataFrame using a fitted Encoding.

    Input:
        df: any DataFrame with the three categorical columns.
        encoding: produced by encode_categoricals on training data.

    Output:
        Copy of df with three new int columns: vessel_idx, berth_idx,
        service_idx. Unknown categories receive OOV_INDEX (-1).

    Description:
        Used to prepare both train rows (no OOV expected) and test rows
        (may contain OOV) for the model. The model layer treats OOV as a
        zero group offset so predictions fall back to the global intercept.
    """
    out = df.copy()
    out["vessel_idx"] = out[vessel_col].map(encoding.vessel).fillna(OOV_INDEX).astype(int)
    out["berth_idx"] = out[berth_col].map(encoding.berth).fillna(OOV_INDEX).astype(int)
    out["service_idx"] = out[service_col].map(encoding.service).fillna(OOV_INDEX).astype(int)
    return out


@dataclass(frozen=True)
class CovariateScaler:
    """
    Mean and std for each continuous covariate, fit on training data.

    Attributes:
        means: dict feature -> training mean (used to center test data)
        stds: dict feature -> training std (used to scale test data)
        feature_cols: ordered list of standardized covariate column names
        beta_names: list of pretty names for the beta coefficients (parallel to feature_cols)

    Description:
        Stored alongside the Encoding so that test-time predictions and
        per-cell posterior predictive use the exact same standardization.
    """

    means: Dict[str, float]
    stds: Dict[str, float]
    feature_cols: list
    beta_names: list


def add_covariates(df: pd.DataFrame, scaler: "CovariateScaler | None" = None) -> tuple[pd.DataFrame, "CovariateScaler"]:
    """
    Compute the M1 covariate columns and standardize them on the training distribution.

    Input:
        df: DataFrame with TRG, Calado diff, atraque_hour/dow/year columns.
        scaler: if None, fit means/stds on this df (training case). If
                provided, reuse them (test case). Reusing avoids leakage.

    Output:
        (df_with_covariates, scaler) where df_with_covariates has new
        columns: log_trg, abs_calado_diff, hour_sin, hour_cos, dow_sin,
        dow_cos, year_trend, plus their standardized z_* counterparts
        which are what the model actually consumes.

    Description:
        Cyclic encodings turn 24-hour and 7-day clocks into sin/cos pairs
        so the linear model captures circular structure. Year is encoded
        as a linear trend centered on 2022 (the data midpoint). All
        continuous features are standardized to mean 0 / sd 1 on the
        training distribution, so beta priors of Normal(0, 0.5) are
        directly interpretable.
    """
    out = df.copy()

    out["log_trg"] = np.log(out["TRG"].clip(lower=1.0))
    out["abs_calado_diff"] = np.abs(out["Calado diff"])
    out["hour_sin"] = np.sin(2 * np.pi * out["atraque_hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["atraque_hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["atraque_dayofweek"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["atraque_dayofweek"] / 7.0)
    out["year_trend"] = out["atraque_year"].astype(float) - 2022.0

    feature_cols = ["log_trg", "abs_calado_diff", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "year_trend"]
    beta_names = ["beta_log_trg", "beta_abs_calado_diff", "beta_hour_sin", "beta_hour_cos",
                  "beta_dow_sin", "beta_dow_cos", "beta_year_trend"]

    if scaler is None:
        means = {c: float(out[c].mean()) for c in feature_cols}
        stds = {c: float(out[c].std(ddof=1)) for c in feature_cols}
        # Guard against zero variance (constant column).
        stds = {c: (s if s > 1e-9 else 1.0) for c, s in stds.items()}
        scaler = CovariateScaler(means=means, stds=stds, feature_cols=feature_cols, beta_names=beta_names)

    for c in feature_cols:
        out[f"z_{c}"] = (out[c] - scaler.means[c]) / scaler.stds[c]

    return out, scaler


def add_log_target(df: pd.DataFrame, target_col: str = "service_time_hours") -> pd.DataFrame:
    """
    Append a log-transformed target column.

    Input:
        df: DataFrame containing the raw service_time_hours column.
        target_col: name of the raw positive target.

    Output:
        Copy of df with new column 'log_service_time' = log(service_time_hours).

    Description:
        The Lognormal likelihood is implemented as Normal on log(svc), so
        the model receives log(target). Centralized here so train/test/PPC
        all use the exact same transform.
    """
    out = df.copy()
    out["log_service_time"] = np.log(out[target_col].to_numpy())
    return out


def prepare(
    csv_path: str | Path,
    train_year_max: int = 2024,
    target_col: str = "service_time_hours",
    vessel_col: str = "Tipo nave (agrupado)",
    berth_col: str = "Sitio",
    service_col: str = "Servicio",
    with_covariates: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, Encoding, "CovariateScaler | None"]:
    """
    End-to-end pipeline: load -> split -> fit encoding -> apply -> add log target.

    Input:
        csv_path: path to training_dataset_bhm.csv.
        train_year_max / target_col / *_col: passed through to sub-steps.
        with_covariates: if True (default), also compute and append
            standardized covariate columns (z_log_trg, z_abs_calado_diff,
            z_hour_sin/cos, z_dow_sin/cos, z_year_trend). M0 ignores them
            but they cost almost nothing to compute and let any downstream
            model use them without re-running the pipeline.

    Output:
        (train_df, test_df, encoding, scaler) where both DataFrames already
        contain vessel_idx, berth_idx, service_idx, log_service_time, and
        (if with_covariates=True) z_* covariate columns. scaler is None if
        with_covariates=False.

    Description:
        Convenience wrapper used by fit.py and tests so callers do not need
        to remember the order of operations.
    """
    df = load_bhm(csv_path)
    train_df, test_df = time_split(df, train_year_max=train_year_max)
    encoding = encode_categoricals(
        train_df, vessel_col=vessel_col, berth_col=berth_col, service_col=service_col
    )
    train_df = apply_encoding(train_df, encoding, vessel_col, berth_col, service_col)
    test_df = apply_encoding(test_df, encoding, vessel_col, berth_col, service_col)
    train_df = add_log_target(train_df, target_col=target_col)
    test_df = add_log_target(test_df, target_col=target_col)
    scaler = None
    if with_covariates:
        train_df, scaler = add_covariates(train_df, scaler=None)
        test_df, _ = add_covariates(test_df, scaler=scaler)
    return train_df, test_df, encoding, scaler

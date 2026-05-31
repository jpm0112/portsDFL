"""
Generate per-cell posterior predictive distributions of service time.

For each (vessel type, berth, service) combination of interest, draws N
posterior predictive samples and writes them to a long-format parquet
file. This file is the consumable for the downstream DFL berth allocation
model: each cell becomes an empirical distribution.

For models with covariates (M1), per-cell prediction needs to fix values
for the covariates. We use the *training-set within-cell mean* of each
standardized covariate when available, and the global training mean (zero
on the standardized scale) for unobserved cells. This makes each per-cell
distribution a "typical" prediction for that cell.
"""

from __future__ import annotations

import os
# Set before importing pymc, in case a JAX-backed sampler is used.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import itertools
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from .data_prep import CovariateScaler, Encoding, prepare
from .models.registry import build, set_predict_data


def build_cells_dataframe(
    train_df: pd.DataFrame,
    encoding: Encoding,
    scaler: CovariateScaler | None,
    include_unobserved: bool,
) -> pd.DataFrame:
    """
    Construct the table of cells to predict, with covariate fill-ins where needed.

    Input:
        train_df: training rows.
        encoding: provides level lists for the Cartesian product.
        scaler: if not None, the returned DataFrame includes z_<feature>
                columns set to the within-cell mean (or global mean when
                the cell is unobserved). M0/M2/M3/M4 ignore them.
        include_unobserved: include all Cartesian-product cells, not just observed ones.

    Output:
        DataFrame with columns: vessel, berth, service, vessel_idx,
        berth_idx, service_idx, n_train, plus z_* columns when scaler is given.
    """
    observed = (
        train_df.groupby(["vessel_idx", "berth_idx", "service_idx"]).size()
        .reset_index(name="n_train")
    )
    if include_unobserved:
        # All combinations of the three factors' integer levels.
        product = list(itertools.product(
            range(encoding.n_vessel), range(encoding.n_berth), range(encoding.n_service),
        ))
        cells = pd.DataFrame(product, columns=["vessel_idx", "berth_idx", "service_idx"])
        # Left merge keeps every cell; cells absent from training get NaN n_train.
        cells = cells.merge(observed, on=["vessel_idx", "berth_idx", "service_idx"], how="left")
        cells["n_train"] = cells["n_train"].fillna(0).astype(int)
    else:
        cells = observed.copy()  # only predict for cells seen in training

    # Reverse lookups index -> name.
    inv_v = {i: v for v, i in encoding.vessel.items()}
    inv_b = {i: v for v, i in encoding.berth.items()}
    inv_s = {i: v for v, i in encoding.service.items()}
    cells["vessel"] = cells["vessel_idx"].map(inv_v)
    cells["berth"] = cells["berth_idx"].map(inv_b)
    cells["service"] = cells["service_idx"].map(inv_s)

    if scaler is not None:
        z_cols = [f"z_{c}" for c in scaler.feature_cols]
        # Within-cell training means (standardized scale); fall back to 0
        # (the global mean on the standardized scale) for unobserved cells.
        cell_means = (
            train_df.groupby(["vessel_idx", "berth_idx", "service_idx"])[z_cols].mean()
            .reset_index()
        )
        cells = cells.merge(cell_means, on=["vessel_idx", "berth_idx", "service_idx"], how="left")
        for c in z_cols:
            cells[c] = cells[c].fillna(0.0)  # unobserved cell -> global mean (0 on z-scale)

    # Sort for stable, reproducible row order.
    return cells.sort_values(["vessel", "berth", "service"]).reset_index(drop=True)


def predict_cells(
    model_key: str,
    train_df: pd.DataFrame,
    encoding: Encoding,
    scaler: CovariateScaler | None,
    idata: az.InferenceData,
    cells: pd.DataFrame,
    n_draws: int,
    priors_kwargs: dict,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Draw posterior predictive samples per cell using the registered model.

    Output: long-format DataFrame [vessel, berth, service, n_train, draw, svc_hours].
    """
    # Rebuild the exact same model that produced `idata` so the saved posterior
    # samples line up with its parameters.
    model = build(model_key, train_df=train_df, encoding=encoding, scaler=scaler, **priors_kwargs)
    # Swap the model's input data from training rows to the cells we predict for.
    set_predict_data(model, cells, encoding, scaler)

    with model:
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["log_y_obs"], random_seed=random_seed, progressbar=True
        )

    # Flatten chain+draw into one "sample" axis.
    samples_log = (
        ppc.posterior_predictive["log_y_obs"].stack(sample=("chain", "draw")).values
    )  # (n_cells, n_total_samples)
    n_total = samples_log.shape[1]
    rng = np.random.default_rng(random_seed)
    # Sample WITH replacement only when we want more draws than exist.
    col_idx = rng.choice(n_total, size=n_draws, replace=(n_draws > n_total))
    samples_log = samples_log[:, col_idx]

    samples_hours = np.exp(samples_log)

    # Reshape (n_cells, n_draws) into a long/tidy table with one row per (cell, draw).
    cell_meta = cells[["vessel", "berth", "service", "n_train"]].reset_index(drop=True)
    repeated = cell_meta.loc[cell_meta.index.repeat(n_draws)].reset_index(drop=True)
    repeated["draw"] = np.tile(np.arange(n_draws), len(cells))
    # flatten() reads row-major, matching the repeat/tile ordering above.
    repeated["svc_hours"] = samples_hours.flatten()
    return repeated


def write_predictive(config_path: str | Path, trace_path: str | Path | None = None) -> Path:
    """
    End-to-end: load config, rebuild data + model, sample PPC, write parquet.
    """
    import yaml  # imported here (not at top) so the module loads even without PyYAML

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # config lives in <base>/configs/<file>.yaml, so base = project root.
    base = Path(config_path).resolve().parent.parent
    model_key = cfg["model_key"]

    train_df, _test_df, encoding, scaler = prepare(
        csv_path=base / cfg["data"]["path"],
        train_year_max=cfg["data"]["train_year_max"],
        target_col=cfg["data"]["target"],
        vessel_col=cfg["data"]["vessel_col"],
        berth_col=cfg["data"]["berth_col"],
        service_col=cfg["data"]["service_col"],
        with_covariates=True,
    )

    # Use the explicit override if given, else the path from the config.
    trace_path = Path(trace_path) if trace_path is not None else base / cfg["output"]["trace"]
    idata = az.from_netcdf(str(trace_path))

    cells = build_cells_dataframe(
        train_df=train_df, encoding=encoding, scaler=scaler,
        include_unobserved=cfg["output"]["predict_unobserved_cells"],
    )

    long_df = predict_cells(
        model_key=model_key,
        train_df=train_df, encoding=encoding, scaler=scaler,
        idata=idata, cells=cells,
        n_draws=cfg["output"]["n_predictive_draws"],
        priors_kwargs=dict(cfg["priors"]),
        random_seed=cfg["sampler"]["random_seed"],
    )

    out_path = base / cfg["output"]["posterior_samples"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(out_path, index=False)
    print(f"[{model_key}] Per-cell posterior predictive written to: {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Write per-cell posterior predictive samples.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--trace", default=None, help="Optional override for trace path.")
    args = p.parse_args()
    write_predictive(args.config, args.trace)

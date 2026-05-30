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

# `from __future__ import annotations` makes every type hint in this file be
# stored as plain text instead of being evaluated. That lets us write hints like
# `CovariateScaler | None` (the `|` "or" syntax) even on older Python versions.
from __future__ import annotations

import os
# Tell the JAX backend (used by some PyMC samplers) to use 64-bit floats.
# `setdefault` only sets the environment variable if it is not already set,
# so it never overrides a value the user provided. Done before importing pymc.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import itertools  # itertools.product() builds the Cartesian product of levels
from pathlib import Path  # Path is an object-oriented file-path type

import arviz as az  # ArviZ: diagnostics + storage for Bayesian results (InferenceData)
import numpy as np
import pandas as pd
import pymc as pm  # PyMC: defines and samples the Bayesian model

# Leading-dot imports are "relative imports": they pull from sibling modules
# inside this same package (the `src` folder), not from installed libraries.
from .data_prep import CovariateScaler, Encoding, prepare
from .models.registry import build, set_predict_data


# `def name(args) -> ReturnType:` defines a function; the part after `->` is the
# return type hint (here a pandas DataFrame). Each `arg: Type` annotates a
# parameter's expected type. `scaler: CovariateScaler | None` means scaler is
# either a CovariateScaler object OR the value None (no scaler / no covariates).
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
    # Count how many training rows fall in each (vessel, berth, service) cell.
    # groupby(...).size() counts rows per group; reset_index turns the grouped
    # result back into a normal DataFrame with a new column named "n_train".
    observed = (
        train_df.groupby(["vessel_idx", "berth_idx", "service_idx"]).size()
        .reset_index(name="n_train")
    )
    if include_unobserved:
        # Build every possible cell = all combinations of the three factors'
        # integer levels. itertools.product is the "for-all-combinations" loop;
        # range(n) yields 0..n-1, the trained level indices for each factor.
        product = list(itertools.product(
            range(encoding.n_vessel), range(encoding.n_berth), range(encoding.n_service),
        ))
        cells = pd.DataFrame(product, columns=["vessel_idx", "berth_idx", "service_idx"])
        # Left-merge keeps every cell; cells absent from training get NaN n_train.
        cells = cells.merge(observed, on=["vessel_idx", "berth_idx", "service_idx"], how="left")
        cells["n_train"] = cells["n_train"].fillna(0).astype(int)  # unobserved -> 0
    else:
        cells = observed.copy()  # only predict for cells seen in training

    # Build reverse lookups index -> name. This is a "dict comprehension":
    # for each (name v, index i) pair in the encoding dict, store {i: v}.
    inv_v = {i: v for v, i in encoding.vessel.items()}
    inv_b = {i: v for v, i in encoding.berth.items()}
    inv_s = {i: v for v, i in encoding.service.items()}
    # .map(dict) translates each integer index back to its human-readable name.
    cells["vessel"] = cells["vessel_idx"].map(inv_v)
    cells["berth"] = cells["berth_idx"].map(inv_b)
    cells["service"] = cells["service_idx"].map(inv_s)

    if scaler is not None:
        # `f"z_{c}"` is an f-string: it inserts the value of c into the text,
        # so feature "log_trg" becomes the column name "z_log_trg". This list
        # comprehension builds the list of standardized covariate column names.
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

    # Sort for stable, reproducible row order; reset_index gives a clean 0..N-1 index.
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
    # Rebuild the exact same model object used during fitting. `**priors_kwargs`
    # unpacks a dict of keyword arguments into the call (e.g. {"beta_sd": 0.5}
    # becomes beta_sd=0.5). The model must match the one that produced `idata`
    # so the saved posterior samples line up with its parameters.
    model = build(model_key, train_df=train_df, encoding=encoding, scaler=scaler, **priors_kwargs)
    # Swap the model's input data from the training rows to the cells we want
    # to predict for (in place). After this, the model's observations have the
    # shape/identity of `cells`, not the training set.
    set_predict_data(model, cells, encoding, scaler)

    # `with model:` is a context manager: PyMC needs an "active model" context
    # for the call below, and the with-block makes `model` that active model.
    with model:
        # Draw posterior predictive samples: for each posterior parameter draw,
        # simulate new outcomes from the likelihood (log_y_obs) for every cell.
        # This propagates full parameter uncertainty into the predictions.
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["log_y_obs"], random_seed=random_seed, progressbar=True
        )

    # The PPC result is an xarray with dims (chain, draw, obs). `.stack` merges
    # the chain and draw dims into one "sample" dim, then `.values` extracts a
    # plain NumPy array. Resulting shape: (n_cells, n_total_samples).
    samples_log = (
        ppc.posterior_predictive["log_y_obs"].stack(sample=("chain", "draw")).values
    )  # (n_cells, n_total_samples)
    n_total = samples_log.shape[1]  # number of available posterior-predictive draws
    rng = np.random.default_rng(random_seed)  # seeded RNG -> reproducible subsample
    # Pick n_draws columns. Sample WITH replacement only when we want more draws
    # than exist (n_draws > n_total); otherwise sample distinct columns.
    col_idx = rng.choice(n_total, size=n_draws, replace=(n_draws > n_total))
    samples_log = samples_log[:, col_idx]  # keep all cells (rows), selected draws (cols)

    samples_hours = np.exp(samples_log)  # undo the log transform: log-hours -> hours

    # Reshape from a (n_cells, n_draws) matrix into a long/tidy table with one
    # row per (cell, draw). index.repeat(n_draws) duplicates each cell row
    # n_draws times: [cell0]*n_draws, [cell1]*n_draws, ...
    cell_meta = cells[["vessel", "berth", "service", "n_train"]].reset_index(drop=True)
    repeated = cell_meta.loc[cell_meta.index.repeat(n_draws)].reset_index(drop=True)
    # np.tile repeats [0..n_draws-1] once per cell, giving each cell a 0..n_draws-1 draw id.
    repeated["draw"] = np.tile(np.arange(n_draws), len(cells))
    # .flatten() reads row-major (cell 0's draws, then cell 1's, ...), which
    # matches the repeat/tile ordering above so values align to the right cell.
    repeated["svc_hours"] = samples_hours.flatten()
    return repeated


def write_predictive(config_path: str | Path, trace_path: str | Path | None = None) -> Path:
    """
    End-to-end: load config, rebuild data + model, sample PPC, write parquet.
    """
    import yaml  # imported here (not at top) so the module loads even without PyYAML

    # `with open(...) as f:` opens the file and guarantees it is closed when the
    # block ends (even on error). safe_load parses YAML text into a plain dict.
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # .resolve() makes an absolute path; .parent.parent climbs two folders up
    # (config lives in <base>/configs/<file>.yaml, so two .parent gets <base>).
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
    # `A if cond else B` is Python's inline (ternary) conditional expression.
    # `base / cfg[...]` joins paths with the / operator (Path overloads it).
    trace_path = Path(trace_path) if trace_path is not None else base / cfg["output"]["trace"]
    # Load the saved posterior (the fitted model's MCMC draws) from a NetCDF file.
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
    # Create the output directory (and any missing parents); don't error if it
    # already exists. parents=True is like `mkdir -p`.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(out_path, index=False)  # write the tidy table to Parquet
    print(f"[{model_key}] Per-cell posterior predictive written to: {out_path}")
    return out_path


# `if __name__ == "__main__":` runs the block below only when this file is
# executed directly (e.g. `python -m src.posterior_predictive`), NOT when it is
# imported by another module. It's the script's command-line entry point.
if __name__ == "__main__":
    import argparse  # standard library command-line argument parser

    p = argparse.ArgumentParser(description="Write per-cell posterior predictive samples.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--trace", default=None, help="Optional override for trace path.")
    args = p.parse_args()  # read the flags the user typed on the command line
    write_predictive(args.config, args.trace)

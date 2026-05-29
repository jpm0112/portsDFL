"""
Held-out evaluation for the M0 BHM and baseline comparators.

Pipeline:
    1. Load trace + reconstruct model + sample predictive on test rows.
    2. Compute scoring rules and calibration metrics on hours-scale.
    3. Compare against NoPoolingBaseline and FullPoolingBaseline.
    4. Slice metrics by training-set cell size (n_train) so the
       partial-pooling story shows up directly.

Outputs:
    - dict of overall metrics per model
    - DataFrame of per-row predictions and CDFs (for calibration plots)
    - DataFrame of per-cell metric slices
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import yaml

from .baselines import FullPoolingBaseline, NoPoolingBaseline
from .data_prep import prepare
from .models.registry import build, set_predict_data


@dataclass
class EvalArtifacts:
    """
    Bundle of evaluation outputs returned by run_evaluation.

    Fields:
        model_key: which registered model these results are for.
        metrics: dict[model_name -> dict[metric -> float]]
        per_row: DataFrame with one row per test obs and columns
                 [model, vessel, berth, service, n_train, y, y_hat_median,
                  pit, lpd, in_50, in_90, abs_err, sq_err]
        per_size_bin: DataFrame with metrics aggregated by n_train bucket.
        idata: posterior trace (returned for use by figures.py).
        train_df, test_df: DataFrames returned for downstream figures.
        encoding: Encoding used at fit time.
        scaler: CovariateScaler (or None) used at fit time.
        priors: dict of priors used to rebuild the model for prediction.
    """

    model_key: str
    metrics: dict
    per_row: pd.DataFrame
    per_size_bin: pd.DataFrame
    idata: az.InferenceData
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    encoding: object
    scaler: object
    priors: dict


# Cell-size bins used to slice metrics. Designed to expose the
# partial-pooling story: very sparse, sparse, medium, dense.
N_TRAIN_BINS = [-0.5, 0.5, 4.5, 19.5, 99.5, np.inf]
N_TRAIN_LABELS = ["unseen (n=0)", "very sparse (1-4)", "sparse (5-19)", "medium (20-99)", "dense (100+)"]


def _compute_n_train_per_cell(train_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of (vessel_idx, berth_idx, service_idx) -> n_train."""
    return (
        train_df.groupby(["vessel_idx", "berth_idx", "service_idx"])
        .size()
        .reset_index(name="n_train")
    )


def _attach_n_train(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Attach n_train to each test row by merging on the cell key."""
    counts = _compute_n_train_per_cell(train_df)
    out = test_df.merge(counts, on=["vessel_idx", "berth_idx", "service_idx"], how="left")
    out["n_train"] = out["n_train"].fillna(0).astype(int)
    return out


def _bhm_predictive_samples(
    model_key: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    encoding,
    scaler,
    idata: az.InferenceData,
    priors: dict,
    n_draws: int,
    random_seed: int,
) -> np.ndarray:
    """
    Draw posterior predictive samples for the test rows under any registered BHM.

    Output: ndarray (n_test, n_draws) in service_time_hours.
    """
    model = build(model_key, train_df=train_df, encoding=encoding, scaler=scaler, **priors)
    set_predict_data(model, test_df, encoding, scaler)
    with model:
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["log_y_obs"], random_seed=random_seed, progressbar=False
        )
    log_samples = (
        ppc.posterior_predictive["log_y_obs"].stack(sample=("chain", "draw")).values
    )  # (n_test, n_total_samples)
    n_total = log_samples.shape[1]
    rng = np.random.default_rng(random_seed)
    cols = rng.choice(n_total, size=n_draws, replace=(n_draws > n_total))
    return np.exp(log_samples[:, cols])


def _scoring_rules(samples: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """
    Per-row scoring rules and calibration quantities.

    Input:
        samples: (n_test, n_draws) predictive samples in svc_hours.
        y: (n_test,) observed svc_hours.

    Output:
        dict with keys:
            y_hat_median, lpd (log predictive density), pit (probability
            integral transform), abs_err, sq_err, in_50, in_90, crps.

    Description:
        - LPD is approximated empirically via a kernel density estimate of
          the predictive samples evaluated at the truth, on the log scale.
          Using log scale matches the Lognormal likelihood and avoids
          numerical issues at the truth tails.
        - PIT is the empirical CDF of samples evaluated at y.
        - CRPS uses the energy-distance form (Gneiting & Raftery 2007):
          CRPS = E|X-y| - 0.5 * E|X-X'|
          which is unbiased to ordering and easy to compute from samples.
    """
    n_test, n_draws = samples.shape
    y_hat_median = np.median(samples, axis=1)

    # PIT: P(X <= y) under the predictive distribution, empirical estimate.
    pit = (samples <= y[:, None]).mean(axis=1)

    in_50 = ((np.quantile(samples, 0.25, axis=1) <= y) & (y <= np.quantile(samples, 0.75, axis=1))).astype(float)
    in_90 = ((np.quantile(samples, 0.05, axis=1) <= y) & (y <= np.quantile(samples, 0.95, axis=1))).astype(float)

    abs_err = np.abs(y_hat_median - y)
    sq_err = (y_hat_median - y) ** 2

    # CRPS via the sample energy form. O(n_test * n_draws); fine for our sizes.
    term1 = np.mean(np.abs(samples - y[:, None]), axis=1)
    # E|X - X'|: take pairs of samples; computed efficiently as
    # sum_i |x_i - x_j| / (n^2). Use sorted form for speed.
    sorted_s = np.sort(samples, axis=1)
    weights = np.arange(1, n_draws + 1)
    # Mean absolute pairwise difference for each row.
    term2 = (2 * (weights * sorted_s).sum(axis=1) - (n_draws + 1) * sorted_s.sum(axis=1)) / (n_draws ** 2)
    crps = term1 - 0.5 * term2

    # Log predictive density on log scale: log p(log y) is Normal-density-like
    # for samples (after taking log). Use a Gaussian KDE proxy: fit to
    # log(samples) and evaluate at log(y). This is robust and matches the
    # natural scale of the model.
    log_samples = np.log(samples)
    log_y = np.log(y)
    # Plug-in normal density with sample mean and sd; equivalent to
    # assuming the predictive on log-scale is approximately Normal, which
    # is exactly what M0 assumes. This avoids KDE bandwidth choices.
    mu_log = log_samples.mean(axis=1)
    sd_log = log_samples.std(axis=1, ddof=1)
    sd_log = np.where(sd_log < 1e-6, 1e-6, sd_log)
    # log Normal pdf on log y, plus -log(y) Jacobian to put on hours scale.
    lpd_hours = (
        -0.5 * np.log(2 * np.pi * sd_log ** 2)
        - 0.5 * ((log_y - mu_log) / sd_log) ** 2
        - log_y
    )

    return {
        "y_hat_median": y_hat_median,
        "lpd": lpd_hours,
        "pit": pit,
        "abs_err": abs_err,
        "sq_err": sq_err,
        "in_50": in_50,
        "in_90": in_90,
        "crps": crps,
    }


def _aggregate(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    """Reduce per-row metric arrays to scalars: means + RMSE."""
    return {
        "mean_lpd": float(metrics["lpd"].mean()),
        "rmse": float(np.sqrt(metrics["sq_err"].mean())),
        "mae": float(metrics["abs_err"].mean()),
        "coverage_50": float(metrics["in_50"].mean()),
        "coverage_90": float(metrics["in_90"].mean()),
        "crps": float(metrics["crps"].mean()),
    }


def run_evaluation(config_path: str | Path, n_draws: int = 2000) -> EvalArtifacts:
    """
    End-to-end evaluation: load trace, predict on test, score, compare baselines.

    Input:
        config_path: YAML config (must point at the same data the trace was fit on).
        n_draws: number of predictive draws per test row per model.

    Output:
        EvalArtifacts (see dataclass).

    Description:
        Computes metrics for three predictive models on the held-out 2025
        rows: M0 (partial pooling), no-pooling, full-pooling. Returns
        per-row predictions for plotting and per-bucket aggregates for the
        partial-pooling story.
    """
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    model_key = cfg["model_key"]
    base = Path(config_path).resolve().parent.parent

    train_df, test_df, encoding, scaler = prepare(
        csv_path=base / cfg["data"]["path"],
        train_year_max=cfg["data"]["train_year_max"],
        target_col=cfg["data"]["target"],
        vessel_col=cfg["data"]["vessel_col"],
        berth_col=cfg["data"]["berth_col"],
        service_col=cfg["data"]["service_col"],
        with_covariates=True,
    )
    test_df = _attach_n_train(test_df, train_df)
    y_test = test_df[cfg["data"]["target"]].to_numpy()

    idata = az.from_netcdf(str(base / cfg["output"]["trace"]))
    priors = dict(cfg["priors"])  # whatever the config provides goes through

    rng = np.random.default_rng(cfg["sampler"]["random_seed"])

    # Predictive samples per model.
    samples_m0 = _bhm_predictive_samples(
        model_key, train_df, test_df, encoding, scaler, idata, priors, n_draws=n_draws,
        random_seed=cfg["sampler"]["random_seed"],
    )
    nopool = NoPoolingBaseline(train_df)
    samples_nopool = nopool.predictive_samples(test_df, n_draws=n_draws, rng=rng)
    fullpool = FullPoolingBaseline(train_df)
    samples_full = fullpool.predictive_samples(test_df, n_draws=n_draws, rng=rng)

    # Per-row metrics for each model.
    rows = []
    metrics: dict[str, dict[str, float]] = {}
    bhm_label = model_key  # use the registered key as the row label so multi-model comparisons line up
    for name, samples in [(bhm_label, samples_m0), ("no_pool", samples_nopool), ("full_pool", samples_full)]:
        per = _scoring_rules(samples, y_test)
        agg = _aggregate(per)
        metrics[name] = agg
        for i in range(len(test_df)):
            rows.append(
                {
                    "model": name,
                    "vessel": test_df[cfg["data"]["vessel_col"]].iloc[i],
                    "berth": test_df[cfg["data"]["berth_col"]].iloc[i],
                    "service": test_df[cfg["data"]["service_col"]].iloc[i],
                    "n_train": int(test_df["n_train"].iloc[i]),
                    "y": float(y_test[i]),
                    "y_hat_median": float(per["y_hat_median"][i]),
                    "pit": float(per["pit"][i]),
                    "lpd": float(per["lpd"][i]),
                    "in_50": float(per["in_50"][i]),
                    "in_90": float(per["in_90"][i]),
                    "abs_err": float(per["abs_err"][i]),
                    "crps": float(per["crps"][i]),
                }
            )
    per_row = pd.DataFrame(rows)

    # Slice by n_train bucket.
    per_row["n_train_bin"] = pd.cut(per_row["n_train"], bins=N_TRAIN_BINS, labels=N_TRAIN_LABELS)
    per_size_bin = (
        per_row.groupby(["model", "n_train_bin"], observed=True)
        .agg(
            n_obs=("y", "size"),
            mean_lpd=("lpd", "mean"),
            mae=("abs_err", "mean"),
            crps=("crps", "mean"),
            coverage_50=("in_50", "mean"),
            coverage_90=("in_90", "mean"),
        )
        .reset_index()
    )

    return EvalArtifacts(
        model_key=model_key,
        metrics=metrics,
        per_row=per_row,
        per_size_bin=per_size_bin,
        idata=idata,
        train_df=train_df,
        test_df=test_df,
        encoding=encoding,
        scaler=scaler,
        priors=priors,
    )


if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n-draws", type=int, default=2000)
    args = p.parse_args()

    art = run_evaluation(args.config, n_draws=args.n_draws)
    print("\n=== Overall metrics ===")
    print(json.dumps(art.metrics, indent=2))
    print("\n=== Per cell-size bucket ===")
    print(art.per_size_bin.to_string(index=False))

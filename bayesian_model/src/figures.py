"""
Figure generators for the M0 evaluation report.

Each function takes the artifacts from src.evaluation.run_evaluation plus
an output directory and writes one or more PNGs. Figures are designed to
be self-contained: titles and axis labels make them readable without the
report context, since the user's paper may reuse them as standalone plots.
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from .baselines import _LognormalParams
from .evaluation import EvalArtifacts, _bhm_predictive_samples


# Global matplotlib styling applied to every figure made in this module:
# screen/file resolution and a cleaner look (no top/right axis borders).
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

MODEL_COLORS = {
    "M0_partial": "#1f77b4",
    "no_pool": "#d62728",
    "full_pool": "#7f7f7f",
}
MODEL_LABELS = {
    "M0_partial": "M0 partial pooling",
    "no_pool": "no pooling",
    "full_pool": "full pooling",
}


def _save(fig, out_dir: Path, name: str) -> Path:
    """Save figure to out_dir/name, returning the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)  # free the figure's memory (important when making many plots)
    return path


def _ensure_covariate_cols(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    For models that consume z_* features, attach zeros (the global mean on
    the standardized scale) for any missing column. No-op when scaler is None.
    """
    if scaler is None:
        return df
    out = df.copy()
    for c in scaler.feature_cols:
        col = f"z_{c}"
        if col not in out.columns:
            out[col] = 0.0  # 0.0 on the standardized scale = the feature's global mean
    return out


def figure_ppc_overlay(art: EvalArtifacts, out_dir: Path) -> Path:
    """
    Overlay observed log(svc) KDE with replicated draws from M0 posterior predictive.

    Description:
        Sanity check that M0 captures the marginal distribution of the target.
        If the replicated and observed densities diverge, the likelihood
        family is wrong (e.g., heavy tails not captured by Lognormal).
    """
    train_df = art.train_df
    # Posterior-predictive replicates for the training rows: (n_train_rows, n_draws) in hours.
    samples = _bhm_predictive_samples(
        model_key=art.model_key,
        train_df=train_df, test_df=train_df,
        encoding=art.encoding, scaler=art.scaler,
        idata=art.idata, priors=art.priors,
        n_draws=200, random_seed=1,
    )
    obs_log = np.log(train_df["service_time_hours"].to_numpy())
    rep_log = np.log(samples)

    fig, ax = plt.subplots(figsize=(7, 4))
    # One faint step-histogram per posterior draw; together they form a
    # "spaghetti" cloud of plausible densities to compare to observed.
    for j in range(rep_log.shape[1]):
        ax.hist(rep_log[:, j], bins=60, density=True, histtype="step",
                color="#1f77b4", alpha=0.04, lw=0.6)
    ax.hist(obs_log, bins=60, density=True, histtype="step",
            color="black", lw=2, label="observed")
    # Empty line just to create a clean legend entry for the cloud.
    ax.plot([], [], color="#1f77b4", lw=1.5, alpha=0.7, label="M0 replicates (n=200)")
    ax.set_xlabel("log(service_time_hours)")
    ax.set_ylabel("density")
    ax.set_title("Posterior predictive check (training rows)")
    ax.legend(loc="upper right")
    return _save(fig, out_dir, "ppc_overlay.png")


def figure_pit_histogram(art: EvalArtifacts, out_dir: Path) -> Path:
    """
    PIT histograms per model. A well-calibrated predictive is uniform on [0,1].

    Description:
        U-shape => underdispersed (intervals too narrow); inverted-U =>
        overdispersed; right-skew => systematic underprediction.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, model in zip(axes, ["M0_partial", "no_pool", "full_pool"]):
        pit = art.per_row.loc[art.per_row["model"] == model, "pit"].to_numpy()
        ax.hist(pit, bins=20, range=(0, 1), color=MODEL_COLORS[model], edgecolor="white")
        # Reference line: a perfectly uniform PIT puts len(pit)/20 obs in each bin.
        ax.axhline(len(pit) / 20, ls="--", color="black", lw=1, alpha=0.6, label="uniform")
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlim(0, 1)
        ax.set_xlabel("PIT")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].set_ylabel("count")
    fig.suptitle("PIT histograms (held-out 2025)", y=1.02)
    return _save(fig, out_dir, "pit_histogram.png")


def figure_metric_by_cell_size(art: EvalArtifacts, out_dir: Path) -> Path:
    """
    Side-by-side bars of MAE, CRPS, and coverage-90 across cell-size buckets, per model.

    Description:
        The visual centerpiece: shows that M0 wins on probabilistic scores
        (CRPS) precisely where data is sparse, and matches or beats other
        models on dense cells. Coverage panel shows calibration.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    # Pivot into a grid: rows = cell-size bucket, columns = model, value = metric.
    pivot_mae = art.per_size_bin.pivot(index="n_train_bin", columns="model", values="mae")
    pivot_crps = art.per_size_bin.pivot(index="n_train_bin", columns="model", values="crps")
    pivot_cov = art.per_size_bin.pivot(index="n_train_bin", columns="model", values="coverage_90")

    for ax, df, ylabel, title in [
        (axes[0], pivot_mae, "MAE (hours)", "Point error (lower better)"),
        (axes[1], pivot_crps, "CRPS (hours)", "Probabilistic score (lower better)"),
        (axes[2], pivot_cov, "coverage", "90% interval coverage"),
    ]:
        # Fixed model column order so colors/positions are consistent.
        df = df.reindex(columns=["M0_partial", "no_pool", "full_pool"])
        x = np.arange(len(df.index))
        w = 0.27
        # (i-1)*w shift places the three models side by side, centered on each bucket.
        for i, m in enumerate(df.columns):
            ax.bar(x + (i - 1) * w, df[m].values, width=w, color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in df.index], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if title.startswith("90%"):
            ax.axhline(0.9, color="black", ls="--", lw=1, alpha=0.7, label="nominal 0.90")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Performance by training-set cell size — n_train bucket", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "metrics_by_cell_size.png")


def figure_tau_posteriors(art: EvalArtifacts, out_dir: Path) -> Path:
    """
    Posterior densities of every tau the model exposes.

    Description:
        Larger tau means more variance explained by that grouping. Adapts
        across models: M0/M1/M2/M3 have three taus; M4 also has tau_vb.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    post = art.idata.posterior
    palette = {
        "tau_vessel": "#1f77b4",
        "tau_berth": "#2ca02c",
        "tau_service": "#d62728",
        "tau_vb": "#9467bd",
    }
    for name, color in palette.items():
        # Only plot taus the current model actually defined (M4 adds tau_vb).
        if name in post.data_vars:
            vals = post[name].values.ravel()
            az.plot_kde(vals, ax=ax, label=name, plot_kwargs={"color": color, "lw": 2})
    ax.set_xlabel("tau (log-scale group SD)")
    ax.set_ylabel("density")
    ax.set_title("Posterior of group-level standard deviations")
    ax.legend(loc="upper right")
    return _save(fig, out_dir, "tau_posteriors.png")


def figure_borrowed_strength(art: EvalArtifacts, out_dir: Path, k: int = 6) -> Path:
    """
    For k cells (3 sparse, 3 dense), overlay no-pool MLE vs M0 partial-pool predictive.

    Description:
        Visual proof of partial pooling. Sparse cells get a wide, smooth
        M0 distribution that borrows from the global mean; dense cells get
        a tight M0 distribution close to the local data.
    """
    counts = (
        art.train_df.groupby(["vessel_idx", "berth_idx", "service_idx"]).size()
        .reset_index(name="n").sort_values("n")
    )
    # Pick up to 3 "sparse" cells (1-3 obs) and up to 3 "dense" cells (>=50).
    # The max(1, ...) / min(3, ...) guards keep the requested count valid.
    sparse = counts[counts["n"].between(1, 3)].sample(min(3, max(1, (counts["n"].between(1,3)).sum())), random_state=0)
    dense = counts[counts["n"] >= 50].sample(min(3, max(1, (counts["n"]>=50).sum())), random_state=0)
    chosen = pd.concat([sparse, dense], ignore_index=True)

    fake_df = _ensure_covariate_cols(chosen.copy(), art.scaler)
    samples = _bhm_predictive_samples(
        model_key=art.model_key,
        train_df=art.train_df, test_df=fake_df,
        encoding=art.encoding, scaler=art.scaler,
        idata=art.idata, priors=art.priors,
        n_draws=2000, random_seed=2,
    )

    # Reverse maps (index -> name) for human-readable plot labels.
    inv_v = {i: v for v, i in art.encoding.vessel.items()}
    inv_b = {i: v for v, i in art.encoding.berth.items()}
    inv_s = {i: v for v, i in art.encoding.service.items()}

    # Top row sparse cells, bottom row dense cells.
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=False)
    for i, (_, row) in enumerate(chosen.iterrows()):
        ax = axes[i // 3, i % 3]
        cell = (int(row["vessel_idx"]), int(row["berth_idx"]), int(row["service_idx"]))
        n = int(row["n"])
        cell_train = art.train_df[
            (art.train_df["vessel_idx"] == cell[0])
            & (art.train_df["berth_idx"] == cell[1])
            & (art.train_df["service_idx"] == cell[2])
        ]
        log_y = np.log(cell_train["service_time_hours"].to_numpy())
        m0_log = np.log(samples[i])  # M0 predictive draws for THIS cell

        ax.hist(m0_log, bins=50, density=True, color="#1f77b4", alpha=0.55, label=f"M0 partial-pool")
        for v in log_y:
            ax.axvline(v, color="black", lw=1.0, alpha=0.7)
        # No-pool Lognormal (if n>=2): a Lognormal on svc is a Normal on log(svc),
        # whose MLE is the sample mean/SD of log_y.
        if n >= 2:
            mu, sigma = float(log_y.mean()), float(log_y.std(ddof=1))
            xs = np.linspace(m0_log.min() - 0.5, m0_log.max() + 0.5, 200)
            pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
            ax.plot(xs, pdf, color="#d62728", lw=2, label="no-pool MLE")
        ax.set_title(
            f"{inv_v[cell[0]]} x {inv_b[cell[1]]} x {inv_s[cell[2]][:18]}\n(n_train={n})",
            fontsize=9,
        )
        ax.set_xlabel("log(service_time_hours)")
        ax.set_ylabel("density")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Borrowed strength: M0 vs no-pool for sparse and dense cells", y=1.01)
    fig.tight_layout()
    return _save(fig, out_dir, "borrowed_strength.png")


def figure_tail_probabilities(art: EvalArtifacts, out_dir: Path) -> Path:
    """
    For the K most-populated cells, plot P(svc > 60h) and P(svc > 100h).

    Description:
        These tail probabilities are exactly what a stochastic berth
        scheduling model needs: how likely is the berth blocked beyond
        a planning threshold? Posteriors give them directly.
    """
    # Keep the 15 cells with the most training rows.
    counts = (
        art.train_df.groupby(["vessel_idx", "berth_idx", "service_idx"]).size()
        .reset_index(name="n").sort_values("n", ascending=False).head(15)
    )
    counts = _ensure_covariate_cols(counts, art.scaler)
    samples = _bhm_predictive_samples(
        model_key=art.model_key,
        train_df=art.train_df, test_df=counts,
        encoding=art.encoding, scaler=art.scaler,
        idata=art.idata, priors=art.priors,
        n_draws=4000, random_seed=3,
    )
    # Averaging the boolean (samples > t) over the draws gives the posterior
    # predictive probability per cell.
    p60 = (samples > 60).mean(axis=1)
    p100 = (samples > 100).mean(axis=1)

    inv_v = {i: v for v, i in art.encoding.vessel.items()}
    inv_b = {i: v for v, i in art.encoding.berth.items()}
    inv_s = {i: v for v, i in art.encoding.service.items()}
    # Truncate long service names so the tick label stays readable.
    labels = [f"{inv_v[v]}\n{inv_b[b]} | {inv_s[s][:12]}"
              for v, b, s in zip(counts["vessel_idx"], counts["berth_idx"], counts["service_idx"])]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w/2, p60, width=w, color="#1f77b4", label="P(svc > 60h)")
    ax.bar(x + w/2, p100, width=w, color="#d62728", label="P(svc > 100h)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("posterior predictive probability")
    ax.set_title("Tail probabilities for top-15 most-populated training cells")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, out_dir, "tail_probabilities.png")


def figure_cell_intervals(art: EvalArtifacts, out_dir: Path, k: int = 25) -> Path:
    """
    For k randomly chosen cells (mix of sizes), plot 50% and 90% predictive
    intervals as horizontal whiskers, sorted by predicted median.

    Description:
        Quick visual census of predictive uncertainty across cells. Sparse
        cells will have markedly wider whiskers, justifying the value of
        carrying full distributions into the DFL model.
    """
    counts = (
        art.train_df.groupby(["vessel_idx", "berth_idx", "service_idx"]).size()
        .reset_index(name="n")
    )
    # Sample up to k cells at random; min() guards against asking for more than exist.
    chosen = counts.sample(min(k, len(counts)), random_state=7)
    chosen = _ensure_covariate_cols(chosen, art.scaler)
    samples = _bhm_predictive_samples(
        model_key=art.model_key,
        train_df=art.train_df, test_df=chosen,
        encoding=art.encoding, scaler=art.scaler,
        idata=art.idata, priors=art.priors,
        n_draws=3000, random_seed=4,
    )
    # Per-cell quantiles: q05/q95 are the 90% interval edges, q25/q75 the 50%.
    med = np.median(samples, axis=1)
    q05, q25, q75, q95 = np.quantile(samples, [0.05, 0.25, 0.75, 0.95], axis=1)

    inv_v = {i: v for v, i in art.encoding.vessel.items()}
    inv_b = {i: v for v, i in art.encoding.berth.items()}
    inv_s = {i: v for v, i in art.encoding.service.items()}
    # Labels are built in the same row order as samples/med so they stay aligned.
    labels = [
        f"{inv_v[v]} | {inv_b[b]} | {inv_s[s][:14]} (n={n})"
        for v, b, s, n in zip(chosen["vessel_idx"], chosen["berth_idx"], chosen["service_idx"], chosen["n"])
    ]
    # Sort by predicted median; apply the same order to every array and the labels.
    order = np.argsort(med)
    med, q05, q25, q75, q95 = med[order], q05[order], q25[order], q75[order], q95[order]
    labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 0.32 * len(labels) + 1.5))
    y = np.arange(len(labels))
    # Wide 90% whisker first (faint), then thicker 50% whisker, then median dot on top.
    ax.hlines(y, q05, q95, color="#1f77b4", alpha=0.4, lw=2, label="90% interval")
    ax.hlines(y, q25, q75, color="#1f77b4", alpha=0.95, lw=4, label="50% interval")
    ax.scatter(med, y, color="black", s=14, zorder=5, label="median")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("service_time_hours (posterior predictive)")
    ax.set_title(f"Per-cell predictive intervals ({len(labels)} random cells)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(left=0)
    fig.tight_layout()
    return _save(fig, out_dir, "cell_intervals.png")


def make_all_figures(art: EvalArtifacts, out_dir: Path) -> dict[str, Path]:
    """
    Generate every figure used in the M0 evaluation report.

    Output: dict of figure name -> filesystem path.
    Description: convenience wrapper called from src.report.
    """
    return {
        "ppc_overlay": figure_ppc_overlay(art, out_dir),
        "pit_histogram": figure_pit_histogram(art, out_dir),
        "metrics_by_cell_size": figure_metric_by_cell_size(art, out_dir),
        "tau_posteriors": figure_tau_posteriors(art, out_dir),
        "borrowed_strength": figure_borrowed_strength(art, out_dir),
        "tail_probabilities": figure_tail_probabilities(art, out_dir),
        "cell_intervals": figure_cell_intervals(art, out_dir),
    }

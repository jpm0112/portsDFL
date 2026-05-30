"""
Assemble the M0 evaluation report.

Reads evaluation artifacts (metrics + per-cell-size table + tau posterior
summary) and emits a single Markdown file with embedded image references.
The report is intended to be the human-readable companion to the trace,
diagnostics sidecar, and per-cell predictive parquet.
"""

# See figures.py for what `from __future__ import annotations` does (it makes
# the type hints below behave as plain text). Same idea applies in this file.
from __future__ import annotations

import os
# Ask JAX to compute in 64-bit floats; set only if not already configured.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import json  # for reading the diagnostics sidecar (a JSON file).
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from .evaluation import EvalArtifacts


# `dict[str, dict[str, float]]` hint: a dict whose values are themselves
# dicts of float (model name -> {metric name -> value}).
def _fmt_overall_metrics(metrics: dict[str, dict[str, float]]) -> str:
    """Markdown table of overall held-out metrics by model."""
    rows = []
    header = ["model", "mean LPD", "MAE (h)", "RMSE (h)", "CRPS (h)", "cov 50%", "cov 90%"]
    # Each row is the model name plus its formatted metrics. f"{x:.3f}" formats
    # the number to 3 decimals; ":.2f" to 2 decimals.
    for name, m in metrics.items():
        rows.append([
            name,
            f"{m['mean_lpd']:.3f}",
            f"{m['mae']:.2f}",
            f"{m['rmse']:.2f}",
            f"{m['crps']:.2f}",
            f"{m['coverage_50']:.3f}",
            f"{m['coverage_90']:.3f}",
        ])
    # Build a Markdown table: header row, a "---" separator row, then data rows.
    # " | ".join(list) glues the cells with " | " between them.
    out = "| " + " | ".join(header) + " |\n"
    out += "|" + "|".join(["---"] * len(header)) + "|\n"
    for r in rows:
        out += "| " + " | ".join(r) + " |\n"
    return out


def _fmt_per_size_bin(per_size_bin: pd.DataFrame) -> str:
    """Markdown table of metrics sliced by training-cell-size bucket."""
    df = per_size_bin.copy()  # copy so we don't mutate the caller's DataFrame.
    # `.map(lambda x: ...)` applies a small inline function to every value in
    # the column. `lambda x: f"{x:.3f}"` just formats each number as text.
    df["mean_lpd"] = df["mean_lpd"].map(lambda x: f"{x:.3f}")
    df["mae"] = df["mae"].map(lambda x: f"{x:.2f}")
    df["crps"] = df["crps"].map(lambda x: f"{x:.2f}")
    df["coverage_50"] = df["coverage_50"].map(lambda x: f"{x:.2f}")
    df["coverage_90"] = df["coverage_90"].map(lambda x: f"{x:.2f}")
    return df.to_markdown(index=False)  # render the DataFrame as a Markdown table.


def _summarize_taus(idata: az.InferenceData) -> str:
    """Markdown table of posterior mean / 5% / 95% for the global hyperparameters present."""
    # Names we MIGHT want to report; different models define different subsets.
    candidates = [
        "alpha0", "tau_vessel", "tau_berth", "tau_service", "tau_vb",
        "sigma", "sigma_global", "nu",
    ]
    # List comprehension: keep only candidates that actually exist in this
    # model's posterior (avoids asking az.summary for missing variables).
    available = [v for v in candidates if v in idata.posterior.data_vars]
    # `az.summary` computes posterior mean/SD plus a 90% highest-density
    # interval (hdi_prob=0.9 -> the hdi_5%/hdi_95% columns) per parameter.
    summary = az.summary(idata, var_names=available, hdi_prob=0.9)[
        ["mean", "sd", "hdi_5%", "hdi_95%"]
    ].round(3)
    return summary.to_markdown()


def _build_recommendations(art: EvalArtifacts) -> list[str]:
    """
    Inspect metrics and produce concrete next-iteration recommendations.

    Description:
        Each rule fires only if the corresponding metric crosses a
        threshold, so the punch list is data-driven rather than generic.
    """
    recs: list[str] = []  # we accumulate recommendation strings here.
    m0 = art.metrics[art.model_key]      # metrics for the hierarchical model.
    nopool = art.metrics["no_pool"]      # metrics for the no-pooling baseline.

    # Calibration check: is observed 90% coverage far from the nominal 0.90?
    if abs(m0["coverage_90"] - 0.9) > 0.05:
        if m0["coverage_90"] < 0.85:
            recs.append(
                f"90% coverage is {m0['coverage_90']:.2f} (< 0.85): predictive "
                "intervals are too narrow. Consider switching to LogStudent-T "
                "likelihood or letting sigma vary by vessel type."
            )
        elif m0["coverage_90"] > 0.95:
            recs.append(
                f"90% coverage is {m0['coverage_90']:.2f} (> 0.95): predictive "
                "intervals are too wide. Tighter HalfNormal scale on tau "
                "hyperpriors or adding informative covariates would help."
            )

    # Higher LPD (log predictive density) is better. If the baseline beats M0
    # overall, flag it (the hierarchy should still help on sparse cells).
    if m0["mean_lpd"] < nopool["mean_lpd"]:
        recs.append(
            "Mean LPD on the test set favors the no-pooling baseline overall. "
            "Inspect per-bucket table: M0 should still win on sparse cells. "
            "If not, the partial pooling is over-shrinking; widen tau priors."
        )

    # Boolean masking: each [...] inside the brackets is a True/False Series,
    # `&` combines them row-wise, and the outer df[...] keeps matching rows.
    # `.astype(str).str.contains(...)` finds the "very sparse" size bucket.
    sparse_row = art.per_size_bin[
        (art.per_size_bin["model"] == art.model_key)
        & (art.per_size_bin["n_train_bin"].astype(str).str.contains("very sparse"))
    ]
    if not sparse_row.empty:
        # `.iloc[0]` grabs the first matching row's value by position.
        sparse_lpd = float(sparse_row["mean_lpd"].iloc[0])
        nopool_sparse = art.per_size_bin[
            (art.per_size_bin["model"] == "no_pool")
            & (art.per_size_bin["n_train_bin"].astype(str).str.contains("very sparse"))
        ]
        if not nopool_sparse.empty:
            nopool_sparse_lpd = float(nopool_sparse["mean_lpd"].iloc[0])
            # Positive gain = M0 predicts sparse cells better than no-pooling.
            gain = sparse_lpd - nopool_sparse_lpd
            recs.append(
                f"Partial pooling gains {gain:+.2f} log-density per very-sparse "
                "test obs versus no-pool. This is the headline justification for "
                "the hierarchical structure."
            )

    # M1 (covariates) recommendation: if MAE is still meaningful relative
    # to median observed, suggest adding TRG/draft covariates next.
    median_y = float(np.median(art.test_df["service_time_hours"]))
    # MAE as a fraction of the typical (median) service time; > 40% is "large".
    if m0["mae"] / median_y > 0.4:
        recs.append(
            f"MAE = {m0['mae']:.1f} h is large relative to median test svc "
            f"({median_y:.1f} h). M1 with log(TRG) and |Calado diff| covariates "
            "is the next obvious win."
        )

    # Service-vs-vessel-vs-berth: which tau is biggest? -> actionability tip.
    # A larger tau = that grouping explains more variance in log(svc).
    post = art.idata.posterior
    tau_means = {
        "vessel": float(post["tau_vessel"].mean()),
        "berth": float(post["tau_berth"].mean()),
        "service": float(post["tau_service"].mean()),
    }
    # `max(dict, key=dict.get)` returns the KEY whose value is largest.
    biggest = max(tau_means, key=tau_means.get)
    recs.append(
        "Posterior tau means: " + ", ".join(f"{k}={v:.2f}" for k, v in tau_means.items())
        + f". The {biggest} factor explains the most variance, so it is the "
        "highest-leverage dimension for both prediction and operational decisions."
    )

    return recs


def write_report(
    art: EvalArtifacts,
    figures: dict[str, Path],
    diag_json_path: Path,
    out_path: Path,
) -> Path:
    """
    Compose the markdown report and write it.

    Input:
        art: outputs from src.evaluation.run_evaluation.
        figures: dict of figure name -> path (from src.figures.make_all_figures).
        diag_json_path: path to the diagnostics sidecar emitted by fit.py.
        out_path: where to write the markdown.

    Output:
        The path that was written.

    Description:
        Layout: header / model + data / convergence / overall metrics /
        sliced metrics / calibration / hyperparameters / borrowed strength /
        tails / per-cell intervals / recommendations.
    """
    # Read the diagnostics JSON file into a Python dict.
    diag = json.loads(diag_json_path.read_text(encoding="utf-8"))

    n_train = len(art.train_df)  # number of training rows.
    n_test = len(art.test_df)    # number of held-out test rows.
    enc = art.encoding

    # Figures live under outputs/figures/<model_key>/ so the relative path
    # from outputs/reports/ is ../figures/<model_key>/<name>.
    # `Path(p).name` is just the filename portion of each figure path.
    figs_rel = {k: f"../figures/{art.model_key}/{Path(p).name}" for k, p in figures.items()}

    # We build the report as a list of strings and join them at the very end.
    md: list[str] = []
    md.append(f"# {art.model_key} — evaluation report\n")
    md.append(
        "Lognormal partial-pooling model for vessel service time. "
        "Three crossed main effects (vessel type, berth, service), no covariates, "
        "no interactions. Held-out evaluation on 2025 vessel calls."
    )

    md.append("\n## 1. Data and model\n")
    # The bullets below DESCRIBE the Bayesian model that was fit elsewhere (in
    # fit.py); they are documentation text, not live computation. In words:
    #   - Likelihood: log(service time) is Normal(mu, sigma) -> svc is Lognormal.
    #   - mu is a sum of a global intercept plus group offsets for vessel/berth/
    #     service (partial pooling shrinks each group toward the global mean).
    #   - HalfNormal(0.5) is a positive-only prior on each group SD (tau).
    #   - NUTS is the MCMC sampler PyMC uses to draw from the posterior.
    md.append(
        f"- Train rows: **{n_train}** (years 2020–{art.train_df['atraque_year'].max()})\n"
        f"- Test rows: **{n_test}** (year 2025)\n"
        f"- Levels: **{enc.n_vessel}** vessel types, **{enc.n_berth}** berths, **{enc.n_service}** services\n"
        "- Likelihood: log(svc) ~ Normal(mu, sigma); mu = alpha0 + alpha_vessel + alpha_berth + alpha_service\n"
        "- Non-centered partial pooling: alpha_g[k] = tau_g * z_g[k], tau_g ~ HalfNormal(0.5)\n"
        "- Sampler: PyMC NUTS, 4 chains x (1000 tune + 1000 draws), target_accept=0.95\n"
    )

    md.append("\n## 2. Convergence diagnostics\n")
    # Standard MCMC health checks pulled from the diagnostics sidecar:
    #   - R-hat near 1.0 means chains agree (converged); high values are bad.
    #   - ESS (effective sample size) ~ how many independent draws you really
    #     have; higher is better. "bulk" covers the center, "tail" the extremes.
    #   - divergences flag places NUTS could not explore reliably; want 0.
    md.append(
        f"- max R-hat: **{diag['rhat_max']:.3f}** (threshold {diag['thresholds']['rhat_max']})\n"
        f"- min bulk ESS: **{diag['ess_bulk_min']:.0f}** (threshold {diag['thresholds']['ess_min']})\n"
        f"- min tail ESS: **{diag['ess_tail_min']:.0f}**\n"
        f"- divergences: **{diag['n_divergences']}**\n"
    )

    md.append("\n## 3. Posterior predictive check\n")
    md.append("Marginal log(svc) replicated from the posterior overlays the observed density:\n")
    md.append(f"\n![PPC overlay]({figs_rel['ppc_overlay']})\n")

    # FIX: was hard-coded "661 vessel calls"; use the actual test row count so
    # the heading never disagrees with the real hold-out size.
    md.append(f"\n## 4. Held-out metrics (2025 hold-out, {n_test} vessel calls)\n")
    md.append("\n### Overall\n")
    md.append(_fmt_overall_metrics(art.metrics))

    md.append("\n### Sliced by training-cell size — the partial-pooling story\n")
    md.append(
        "Each test row is bucketed by how many training observations its "
        "(vessel, berth, service) cell had. Partial pooling helps most where "
        "training data is scarce.\n"
    )
    md.append(_fmt_per_size_bin(art.per_size_bin))
    md.append(f"\n![Metrics by cell size]({figs_rel['metrics_by_cell_size']})\n")

    md.append("\n### Calibration (PIT histograms)\n")
    md.append(
        "A well-calibrated predictive distribution produces uniform PIT values. "
        "U-shape => intervals too narrow; inverted-U => too wide.\n"
    )
    md.append(f"\n![PIT]({figs_rel['pit_histogram']})\n")

    md.append("\n## 5. Hyperparameter posteriors\n")
    md.append(
        "Each tau is the standard deviation of group-level offsets. Larger tau "
        "means the corresponding factor explains more variance in log(svc).\n"
    )
    md.append(_summarize_taus(art.idata))
    md.append(f"\n![Tau posteriors]({figs_rel['tau_posteriors']})\n")

    md.append("\n## 6. Borrowed strength\n")
    md.append(
        "For sparse cells (n_train ≤ 3) the model produces a smooth, plausible "
        "predictive distribution by leaning on the global hierarchy. For dense "
        "cells (n_train ≥ 50) M0 tracks the local empirical distribution closely.\n"
    )
    md.append(f"\n![Borrowed strength]({figs_rel['borrowed_strength']})\n")

    md.append("\n## 7. Useful outputs for the DFL berth allocation model\n")
    md.append(
        "Per-cell tail probabilities (P(svc > threshold)) feed directly into "
        "stochastic berth scheduling: the planner needs to know how often a "
        "berth will still be occupied past a planning horizon.\n"
    )
    md.append(f"\n![Tail probabilities]({figs_rel['tail_probabilities']})\n")
    md.append(
        "\nFull predictive intervals per cell (50% and 90%) for a random sample "
        "of cells:\n"
    )
    md.append(f"\n![Cell intervals]({figs_rel['cell_intervals']})\n")

    md.append("\n## 8. Recommended next steps (data-driven)\n")
    # Append each recommendation as a Markdown bullet point.
    for r in _build_recommendations(art):
        md.append(f"- {r}")

    md.append("\n## 9. Artifacts\n")
    md.append(
        f"- Trace: `outputs/traces/{art.model_key}.nc`\n"
        f"- Diagnostics sidecar: `outputs/traces/{art.model_key}.diag.json`\n"
        f"- Per-cell posterior predictive parquet: `outputs/posterior_samples/{art.model_key}_cells.parquet`\n"
        "  (consumed by the downstream DFL berth allocation model)\n"
    )

    # Make sure the output folder exists, then write all sections joined by
    # blank lines into one Markdown file.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path

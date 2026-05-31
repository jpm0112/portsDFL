"""
Cross-model comparison: load every fitted model's trace and evaluation,
rank by held-out predictive performance and PSIS-LOO, and emit a
single summary report.

Usage (from bayesian_model/ folder):
    python -m src.compare_models --configs configs/bhm_baseline.yaml \
        configs/bhm_m1_covariates.yaml configs/bhm_m2_heavytail.yaml \
        configs/bhm_m3_heteroscedastic.yaml configs/bhm_m4_interactions.yaml
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import yaml

from .data_prep import prepare
from .evaluation import _aggregate, _attach_n_train, _scoring_rules, _bhm_predictive_samples


def _maybe_compute_log_likelihood(model_key: str, train_df, encoding, scaler, idata: az.InferenceData, priors: dict) -> az.InferenceData:
    """
    Add log_likelihood group to idata if missing, by recomputing under the
    rebuilt model. Required for az.compare/loo.

    Description:
        We saved traces from pm.sample without idata_kwargs={"log_likelihood": True}
        because those traces are large. Recompute on demand here.
    """
    if "log_likelihood" in idata.groups():
        return idata
    from .models.registry import build  # imported here (not at top) to avoid a circular import
    model = build(model_key, train_df=train_df, encoding=encoding, scaler=scaler, **priors)
    with model:
        # Re-evaluate the per-observation log-likelihood from the stored posterior draws;
        # az.loo/az.compare need this group, which we skipped saving because it's bulky.
        idata = pm.compute_log_likelihood(idata, progressbar=False)
    return idata


def gather_artifacts(config_paths: list[Path], n_draws: int = 1500) -> dict[str, dict]:
    """
    For each config, rebuild data + model, compute held-out predictive
    samples, and bundle metrics + LOO inputs. Returns dict keyed by model_key.

    Output:
        dict model_key -> {
            "config": cfg,
            "metrics": dict of held-out metrics,
            "per_size": DataFrame (held-out metrics sliced by cell-size bin),
            "idata": InferenceData with log_likelihood populated,
            "samples": (n_test, n_draws) predictive samples for plots,
            "y": observed test y,
            "n_train": np.int array per test row,
        }
    """
    out: dict[str, dict] = {}
    for cfg_path in config_paths:
        with open(cfg_path, "r", encoding="utf-8") as f:  # FIX: use `with` so the file handle is closed
            cfg = yaml.safe_load(f)
        model_key = cfg["model_key"]
        # config lives in configs/<file>.yaml, so base = project root (data/, outputs/).
        base = Path(cfg_path).resolve().parent.parent

        # Rebuild the train/test split + encoders exactly as at fit time so column
        # indices and scaling line up with the saved trace.
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
        y_test = test_df[cfg["data"]["target"]].to_numpy()  # held-out service times (hours)

        idata = az.from_netcdf(str(base / cfg["output"]["trace"]))
        priors = dict(cfg["priors"])
        # Ensure log_likelihood is present (recompute if the saved trace omitted it).
        idata = _maybe_compute_log_likelihood(model_key, train_df, encoding, scaler, idata, priors)

        samples = _bhm_predictive_samples(
            model_key, train_df, test_df, encoding, scaler, idata, priors,
            n_draws=n_draws, random_seed=cfg["sampler"]["random_seed"],
        )
        per = _scoring_rules(samples, y_test)
        agg = _aggregate(per)

        # Per-size-bin LPD & CRPS for the comparison plot.
        from .evaluation import N_TRAIN_BINS, N_TRAIN_LABELS
        per_row = pd.DataFrame({
            "n_train": test_df["n_train"].astype(int).to_numpy(),
            "lpd": per["lpd"], "crps": per["crps"], "abs_err": per["abs_err"],
            "in_50": per["in_50"], "in_90": per["in_90"],
        })
        per_row["bin"] = pd.cut(per_row["n_train"], bins=N_TRAIN_BINS, labels=N_TRAIN_LABELS)
        # observed=True keeps only bins that actually occur.
        per_size = (per_row.groupby("bin", observed=True)
                    .agg(n_obs=("lpd", "size"), mean_lpd=("lpd", "mean"),
                         mae=("abs_err", "mean"), crps=("crps", "mean"),
                         coverage_50=("in_50", "mean"), coverage_90=("in_90", "mean"))
                    .reset_index())

        out[model_key] = {
            "config": cfg, "metrics": agg, "per_size": per_size,
            "idata": idata, "samples": samples, "y": y_test,
            "n_train": test_df["n_train"].to_numpy(),
        }
    return out


def make_loo_table(artifacts: dict[str, dict]) -> pd.DataFrame:
    """LOO comparison across all models with log_likelihood populated."""
    idatas = {k: a["idata"] for k, a in artifacts.items()}
    # PSIS-LOO estimates out-of-sample predictive accuracy from the in-sample
    # log-likelihood without refitting; method="stacking" computes model weights.
    # Returned DataFrame is sorted best-first; index is the model name.
    loo = az.compare(idatas, ic="loo", method="stacking")
    return loo


def make_metrics_table(artifacts: dict[str, dict]) -> pd.DataFrame:
    """One-row-per-model held-out metrics summary."""
    rows = []
    for k, a in artifacts.items():
        m = a["metrics"]
        rows.append({"model": k, "mean_lpd": m["mean_lpd"], "rmse": m["rmse"],
                     "mae": m["mae"], "crps": m["crps"],
                     "coverage_50": m["coverage_50"], "coverage_90": m["coverage_90"]})
    # Sort ascending by CRPS (lower is better), so row 0 is the best-CRPS model.
    return pd.DataFrame(rows).sort_values("crps").reset_index(drop=True)


def figure_comparison_metrics(artifacts: dict[str, dict], out_path: Path) -> Path:
    """Bar charts of mean LPD, MAE, CRPS, and coverage_90 across all models."""
    keys = list(artifacts.keys())
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    # lower_better is unused here but documents each metric's direction.
    for ax, metric, title, lower_better in [
        (axes[0], "mean_lpd", "mean LPD (higher better)", False),
        (axes[1], "mae", "MAE hours (lower better)", True),
        (axes[2], "crps", "CRPS hours (lower better)", True),
        (axes[3], "coverage_90", "90% coverage (target 0.9)", False),
    ]:
        vals = [artifacts[k]["metrics"][metric] for k in keys]
        ax.bar(range(len(keys)), vals, color="#1f77b4")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        if metric == "coverage_90":
            ax.axhline(0.9, ls="--", color="black", lw=1, alpha=0.7)  # 0.9 target reference
    fig.suptitle("Held-out 2025 metrics across BHM variants", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return out_path


def figure_per_size_lpd(artifacts: dict[str, dict], out_path: Path) -> Path:
    """Mean LPD by training-cell-size bucket, one line per model."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, a in artifacts.items():
        ps = a["per_size"]
        ax.plot(ps["bin"].astype(str), ps["mean_lpd"], marker="o", label=k)
    ax.set_xlabel("training-cell size bin")
    ax.set_ylabel("mean LPD (higher better)")
    ax.set_title("Probabilistic accuracy by data sparsity")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return out_path


def figure_per_size_crps(artifacts: dict[str, dict], out_path: Path) -> Path:
    """CRPS by training-cell-size bucket, one line per model."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, a in artifacts.items():
        ps = a["per_size"]
        ax.plot(ps["bin"].astype(str), ps["crps"], marker="o", label=k)
    ax.set_xlabel("training-cell size bin")
    ax.set_ylabel("CRPS hours (lower better)")
    ax.set_title("Probabilistic error by data sparsity")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return out_path


def figure_pit_grid(artifacts: dict[str, dict], out_path: Path) -> Path:
    """PIT histograms side-by-side per model."""
    keys = list(artifacts.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.2), sharey=True)
    # With one panel subplots returns a single axis; wrap it so the loop is uniform.
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        samples = artifacts[k]["samples"]  # (n_test, n_draws)
        y = artifacts[k]["y"]              # (n_test,)
        # PIT = predictive CDF at the truth (fraction of draws <= y); uniform if calibrated.
        pit = (samples <= y[:, None]).mean(axis=1)
        ax.hist(pit, bins=20, range=(0, 1), color="#1f77b4", edgecolor="white")
        # Reference line at the height of a perfectly uniform histogram (n / 20 bins).
        ax.axhline(len(pit) / 20, ls="--", color="black", lw=1, alpha=0.6)
        ax.set_title(k, fontsize=9)
        ax.set_xlabel("PIT")
    axes[0].set_ylabel("count")
    fig.suptitle("PIT histograms across BHM variants (held-out 2025)", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return out_path


def write_comparison_report(
    artifacts: dict[str, dict],
    metrics_table: pd.DataFrame,
    loo_table: pd.DataFrame,
    figure_paths: dict[str, Path],
    out_path: Path,
) -> Path:
    """Compose outputs/reports/comparison_summary.md."""
    md: list[str] = []
    md.append("# Cross-model comparison — Bayesian hierarchical service-time models\n")
    md.append(
        "Held-out 2025 evaluation across all M0..M4 variants. The table\n"
        "below summarizes overall metrics; the LOO table ranks models by\n"
        "in-sample expected log predictive density. See per-bucket figures\n"
        "for the partial-pooling story.\n"
    )

    md.append("\n## 1. Held-out metrics (2025, n=661)\n")
    md.append(metrics_table.round(4).to_markdown(index=False))

    md.append("\n## 2. PSIS-LOO ranking (in-sample expected log predictive density)\n")
    md.append(loo_table.round(3).to_markdown())

    md.append("\n## 3. Cross-model figures\n")
    md.append(f"![Overall metrics](../figures/comparison/{figure_paths['metrics'].name})\n")
    md.append(f"![LPD by cell size](../figures/comparison/{figure_paths['lpd_by_size'].name})\n")
    md.append(f"![CRPS by cell size](../figures/comparison/{figure_paths['crps_by_size'].name})\n")
    md.append(f"![PIT grid](../figures/comparison/{figure_paths['pit_grid'].name})\n")

    md.append("\n## 4. Per-model headline notes\n")
    for k, a in artifacts.items():
        m = a["metrics"]
        md.append(f"- **{k}**: LPD {m['mean_lpd']:.3f}, MAE {m['mae']:.2f} h, CRPS {m['crps']:.2f} h, cov90 {m['coverage_90']:.2f}.")

    md.append("\n## 5. Recommendation for downstream DFL\n")
    # loo_table is sorted best-first; metrics_table is sorted ascending by CRPS.
    best_loo_key = loo_table.index[0]
    best_crps_key = metrics_table.iloc[0]["model"]
    md.append(
        f"- LOO favors: **{best_loo_key}**\n"
        f"- Held-out CRPS favors: **{best_crps_key}**\n"
        "- For the DFL berth allocation model, prefer the parquet from the\n"
        "  CRPS leader since CRPS scores the full predictive distribution\n"
        "  on the deployment-time scale (hours, hold-out year)."
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", required=True, help="One or more YAML configs.")
    p.add_argument("--n-draws", type=int, default=1500)
    args = p.parse_args()

    config_paths = [Path(c).resolve() for c in args.configs]
    base = config_paths[0].parent.parent  # project root, inferred from the first config

    artifacts = gather_artifacts(config_paths, n_draws=args.n_draws)
    metrics_table = make_metrics_table(artifacts)
    loo_table = make_loo_table(artifacts)

    fig_dir = base / "outputs" / "figures" / "comparison"
    figure_paths = {
        "metrics": figure_comparison_metrics(artifacts, fig_dir / "overall_metrics.png"),
        "lpd_by_size": figure_per_size_lpd(artifacts, fig_dir / "lpd_by_size.png"),
        "crps_by_size": figure_per_size_crps(artifacts, fig_dir / "crps_by_size.png"),
        "pit_grid": figure_pit_grid(artifacts, fig_dir / "pit_grid.png"),
    }

    out_md = base / "outputs" / "reports" / "comparison_summary.md"
    write_comparison_report(artifacts, metrics_table, loo_table, figure_paths, out_md)

    print("\n=== Held-out metrics ===")
    print(metrics_table.round(3).to_string(index=False))
    print("\n=== LOO ===")
    print(loo_table.round(3).to_string())
    print(f"\nReport: {out_md}")


if __name__ == "__main__":
    main()

# prediction_models

Service-time prediction and Decision-Focused Learning (DFL) for the
**Discrete Berth Allocation Problem (DBAP)** at Puerto de San Antonio,
Chile.

This module is the prediction half of the third paper of the dissertation:
*Decision-Focused Learning for Berth Allocation*. It trains models that
predict vessel service time τ from operational features and then evaluates
whether training the predictor with a *decision-aware* loss yields better
downstream scheduling decisions than a standard MSE-trained predictor.

## What the project answers

Given historical data on Chilean container vessels (arrival, vessel size,
terminal, operator, weather, etc.), can we:

1. Predict service time τ accurately enough to be useful for scheduling?
2. Train the same predictor with a *decision-aware* loss (DFL) and see
   measurable gains on downstream berth-allocation cost compared to the
   standard predict-then-optimize (PtO) recipe?

Pipeline at a glance:

```
features x  ─►  predictor f_θ  ─►  τ̂  ─►  DBAP MILP solver  ─►  schedule z
                                                            │
                            cost under true τ  ◄────────────┘
```

PtO trains f_θ with MSE on (x, τ). DFL trains f_θ with a loss that knows
about the MILP, via Differentiable Black-Box gradients (Pogančić et al.
2020) so the predictor learns where prediction errors actually hurt the
schedule.

## Glossary (DFL literature)

| Term | Meaning |
|---|---|
| **τ** | true service time (target variable) |
| **τ̂** | predicted service time |
| **PtO** | predict-then-optimize: f_θ trained with MSE, MILP solved on τ̂ |
| **DFL** | decision-focused: f_θ trained with a loss that depends on the MILP solution |
| **FI** | full-information optimum: MILP solved on the *true* τ. Lower bound for any prediction-driven schedule. |
| **regret** | cost(decision under τ̂) − cost(FI). Always ≥ 0 (modulo solver gap). |
| **DBB** | Differentiable Black-Box (Pogančić 2020). Gives a surrogate gradient through the MILP via a perturbed-solve interpolation. |
| **SPO+** | Smart Predict-then-Optimize+ (Elmachtoub & Grigas 2022). Convex surrogate, only applies when τ̂ enters the *objective* alone. Not used here because τ̂ also appears in the precedence constraints. |

## Models

Four prediction tiers, each trained under 5-fold CV with Optuna tuning.

| Tier | Model | What it adds | Reference |
|---|---|---|---|
| 1 | **Ridge linear** | A single `nn.Linear(d, 1)` trained with AdamW + MSE + decoupled weight decay. Acts as the simple floor and as the shared backbone for the PtO / DFL comparison so any difference is attributable to the loss, not the architecture. | Hoerl & Kennard 1970; Loshchilov & Hutter 2019 |
| 2 | **RealMLP** | Plain MLP with carefully calibrated defaults (robust scaler, PReLU, parametric numerical embeddings, OneCycle, label smoothing). Matches GBDTs without per-dataset tuning. | Holzmüller et al. NeurIPS 2024 |
| 3 | **TabM** | Parameter-efficient ensemble of MLPs via the BatchEnsemble trick: shared backbone + small per-member rank-1 adapters. Variance reduction at near single-model cost. Forward returns (B, k, 1); we mean over k. | Gorishniy et al. ICLR 2025 |
| 4 | **NODE** | Stack of soft *oblivious* decision trees (CatBoost-style symmetric trees) with dense skip connections and a linear head. Tree inductive bias kept end-to-end differentiable via entmax15 / sparsemoid. | Popov et al. ICLR 2020 |

Plus a **baselines** suite (global mean, per-vessel-type and per-terminal
group means) as a sanity floor.

For DFL, only the Tier 1 head is used so PtO and DFL share the same
architecture and only the loss differs.

## Optimization side: Discrete BAP

Discrete BAP (Cordeau et al. 2005, Bierwirth & Meisel 2010, 2015):

- N vessels assigned to M berths.
- Each vessel has an arrival time aᵢ, a weight wᵢ, a true service time τᵢ.
- Decisions: x[i, b] (assignment), s[i] (start time), z[i, j, b]
  (precedence at berth b).
- Objective: minimize Σᵢ wᵢ (sᵢ + τᵢ), i.e. weighted completion time.
- Big-M precedence constraints couple s, z, τ across vessels at the same
  berth.

Implemented in Pyomo for solver flexibility, with Gurobi as the default
backend. Mutable `Param` for τ enables warm-started re-solves under new
predictions without rebuilding the model. See
`src/ports_dfl/optim/discrete_bap.py`.

## Setup

From this folder, on Windows:

```powershell
.\setup.ps1
.\.venv\Scripts\Activate.ps1
python scripts\check_gpu.py
```

On Linux / WSL:

```bash
./setup.sh
source .venv/bin/activate
python scripts/check_gpu.py
```

`setup.{ps1,sh}` auto-detects the host CUDA version (via `nvidia-smi`) and
installs the matching PyTorch wheels from the official index URL. Falls
back to CPU if no NVIDIA GPU is found. Gurobi requires a license at
`~/gurobi.lic` (academic license is free).

## Run

```powershell
# Sanity floor
python scripts\run_baselines.py

# Each model: train + 5-fold CV + Optuna tuning
python scripts\run_linear.py
python scripts\run_realmlp.py
python scripts\run_tabm.py
python scripts\run_node.py

# DFL end-to-end on the real Discrete BAP (multi-berth scheduling)
python scripts\run_dfl_real_bap.py --n_vessels 8 --n_berths 3 \
    --n_train_instances 60 --n_val_instances 30 --max_epochs 10

# Solver runtime sweep across N = 5..10
python scripts\benchmark_dbb.py

# Aggregate every CV summary and the DBAP demo into one table
python scripts\compare.py

# Generate a multi-page PDF report (results/dfl_report.pdf)
python scripts\build_report.py
```

## Tests

```powershell
pytest -q
```

61 tests cover data loading, encoders, splits, metrics, every model wrapper,
the DBAP MILP, log-target wrapper, and the regret accounting.

## Directory layout

```
prediction_models/
├── src/ports_dfl/
│   ├── config.py            paths, seeds, CV settings, DEVICE
│   ├── data/
│   │   ├── loader.py        load_training_dataset, split_features_target
│   │   ├── encoders.py      target / one-hot / numeric preprocessor
│   │   └── splits.py        5-fold CV split (single-seed, deterministic)
│   ├── metrics/
│   │   └── regression.py    MAE, RMSE, R², MAPE, summarize_folds
│   ├── models/
│   │   ├── base.py          BaseModel ABC (fit / predict / save / load)
│   │   ├── baselines.py     global-mean, group-mean predictors
│   │   ├── linear.py        Ridge as nn.Linear + AdamW (Tier 1)
│   │   ├── realmlp.py       pytabkit RealMLP wrapper (Tier 2)
│   │   ├── tabm.py          tabm package wrapper (Tier 3)
│   │   ├── node.py          pytorch_tabular DenseODSTBlock (Tier 4)
│   │   └── log_target.py    log1p(τ) target wrapper for skewed targets
│   ├── optim/
│   │   └── discrete_bap.py  DBAP MILP (Pyomo + Gurobi), FI helpers
│   ├── train/
│   │   ├── pto.py           generic MSE training loop (AdamW + cosine + AMP)
│   │   └── dfl_blackbox.py  DBB DFL training loop (PyEPO blackboxOpt)
│   └── tuning/
│       ├── runner.py        Optuna study runner
│       └── search_spaces.py per-model parameter search spaces
│
├── scripts/
│   ├── check_gpu.py             prints CUDA / device info
│   ├── run_baselines.py         train + score baselines
│   ├── run_linear.py            CV + Optuna for Ridge
│   ├── run_realmlp.py           CV + Optuna for RealMLP
│   ├── run_tabm.py              CV + Optuna for TabM
│   ├── run_node.py              CV + Optuna for NODE
│   ├── run_dfl_real_bap.py      DFL vs PtO on real DBAP (canonical demo)
│   ├── benchmark_dbb.py         per-solve and per-epoch DFL timing across N
│   ├── compare.py               aggregate every CV summary
│   ├── compare_log_target.py    log1p target vs raw target ablation
│   └── build_report.py          generate results/dfl_report.pdf
│
├── tests/
│   └── (pytest, 61 tests)
│
├── results/
│   ├── baselines/cv_summary.csv
│   ├── linear/cv_summary*.csv
│   ├── realmlp/cv_summary*.csv
│   ├── tabm/cv_summary*.csv
│   ├── node/cv_summary*.csv
│   ├── dfl_real_bap/
│   │   ├── pto_per_instance.csv      per-instance PtO costs / regrets
│   │   ├── dfl_per_instance.csv      same, for DFL
│   │   ├── predictive_summary.csv    fold-0 PtO / DFL MAE / RMSE / R² / MAPE
│   │   ├── decision_summary.csv      regret, makespan, wait, utilization
│   │   ├── dfl_training_trace.csv    per-epoch SPO loss + val regret
│   │   └── config.json               instance descriptor
│   ├── benchmark_dbb.csv             solver runtime sweep
│   ├── comparison.csv                aggregated CV ranking table
│   ├── dfl_report.pdf                9-page PDF results report
│   └── report_preview/               PNG previews of the report pages
│
├── optuna_studies/                   on-disk Optuna study DBs
├── training_dataset.csv              raw input data
├── pyproject.toml                    package + ruff/black/pytest config
├── requirements.txt                  pip dependencies (incl. torch wheel hint)
├── setup.ps1 / setup.sh              CUDA-aware env bootstrap
└── README.md                         this file
```

## Conventions

- One `.py` per model, one test file per model. No monolithic scripts.
- Every model implements `BaseModel` so the CV / Optuna / report code is
  model-agnostic.
- Results CSVs use the same schema across models so `compare.py` and
  `build_report.py` can ingest them uniformly.
- DFL terminology follows the literature (FI, regret, predicted decision)
  rather than the older "oracle" wording.

## Key references

- Cordeau, Laporte, Legato, Moccia (2005). *Models and tabu search heuristics for the berth-allocation problem.*
- Bierwirth & Meisel (2010, 2015). Surveys on berth allocation and quay-crane scheduling.
- Pogančić, Paulus, Musil, Martius, Vlastelica, Rolínek (2020). *Differentiation of blackbox combinatorial solvers.* ICLR.
- Elmachtoub & Grigas (2022). *Smart Predict, then Optimize.* Management Science.
- Mandi, Bucarey, Tschiatschek, Guns (2024). *Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities.* JAIR survey.
- Holzmüller, Grinsztajn, Steinwart (2024). *Better by default: Strong pre-tuned MLPs and boosted trees on tabular data.* NeurIPS.
- Gorishniy et al. (2025). *TabM: Advancing tabular deep learning with parameter-efficient ensembling.* ICLR.
- Popov, Morozov, Babenko (2020). *Neural Oblivious Decision Ensembles for deep learning on tabular data.* ICLR.
- Loshchilov & Hutter (2019). *Decoupled weight decay regularization (AdamW).* ICLR.

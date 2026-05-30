# portsDFL — Decision-Focused Learning for Berth Allocation

Research code for the Port of San Antonio (Chile). The goal: **predict how long each vessel
will occupy a berth (its *service time* τ), feed that into a Berth Allocation Problem (BAP)
MILP that schedules the week, and train the predictor so it produces good *schedules* — not
just low prediction error.** That last idea is **Decision-Focused Learning (DFL)**, compared
against the standard **predict-then-optimize (PtO)** baseline.

## Pipeline

```
 source workbook ─► data_pipeline ─► training_dataset.csv
                                          │
                          ┌───────────────┴───────────────┐
                          ▼                                ▼
                  prediction_models                  bayesian_model
                 (PtO + DFL predict τ̂)            (PyMC service-time
                          │                          models, alt. approach)
                          ▼
                 BAP MILP (the optimizer)  ──►  weekly berth schedule
```

## Repository layout

| Folder | What it is |
|--------|------------|
| **`data_pipeline/`** | Standalone scripts that turn the source Excel workbook into `clean_dataset.csv` and `training_dataset.csv` (cleaning, targets, leakage-safe features). |
| **`prediction_models/`** | The main project: the `ports_dfl` package (PtO + DFL training, models, tuning) and the BAP MILP + weekly planner. Has its own README. |
| **`bayesian_model/`** | A separate PyMC subproject: hierarchical models that predict service time with calibrated uncertainty. |
| **`docs/`** | `project_description.md`, `column_description.md`, plus `literature/` and `meetings/`. |

Start with **`prediction_models/README.md`** for the modelling/optimization detail, and
**`docs/project_description.md`** for the data and problem background.

## Data & reproducibility note

The proprietary port data (`data/`, `external_data/`, the source `*.xlsx`) is **gitignored and
not included** — you supply it locally. Two known gaps in a fresh clone (see
`prediction_models/docs/REVIEW_FINDINGS.md`): the `ports_dfl.data` loader subpackage is missing,
and the model/solver stack (`pyepo`, `torch`, PyMC, Gurobi) must be installed per each
subproject's requirements. The **optimizer layer is self-contained and runs/tests independently**
of the proprietary data.

# optimizers (`bap_optim`)

The optimization layer of portsDFL: the **Discrete Berth Allocation Problem (BAP) MILP** and the
**deterministic weekly planner**. Extracted into its own top-level package so the pipeline stages
— data → prediction → **optimization** — are visible at the repo root, and so the optimizer can be
used/tested independently of the (data-dependent) prediction code.

## Layout
```
optimizers/
├── pyproject.toml
├── src/bap_optim/
│   ├── instance.py          BAPInstance descriptor (numpy-only, no solver dep)
│   ├── discrete_bap.py      the DBAP MILP (Pyomo + Gurobi); PyEPO-compatible
│   ├── berths.py            berth catalog + vessel–berth compatibility matrix
│   ├── weekly_instance.py   pre-solve weekly slicing + synthetic generator
│   ├── schedule.py          schedule assembly + KPIs (numpy-only)
│   ├── classic_bap.py       synthetic instance generator (difficulty knobs)
│   └── __init__.py          lazy API (solver-free names import without PyEPO)
└── tests/                   test_discrete_bap, test_bap_windows, test_weekly_instance
```

## Install / use
```
pip install -e ./optimizers           # exposes the `bap_optim` package
# DFL training also needs PyEPO + a solver:
pip install -e ./optimizers[dfl] gurobipy
```
`prediction_models` imports this package as `bap_optim` (its scripts also add `optimizers/src`
to `sys.path`, so they run without an install).

## Key entry points
- `from bap_optim import DiscreteBAP, BAPInstance` — the MILP.
- `from bap_optim import build_weekly_instance, generate_synthetic_weekly_instance` — instances.
- `from bap_optim import assemble_schedule, compute_kpis` — post-solve reporting.

Dependency-light pieces (`BAPInstance`, berths, weekly builder, schedule) import without PyEPO;
only `DiscreteBAP` pulls in the solver stack. The formulation is documented in
`prediction_models/docs/formulation/`.

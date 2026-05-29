"""ports_dfl: Decision-Focused Learning for the berth allocation problem.

Top-level package. Submodules:
    config   - paths, seeds, CV folds
    data     - loading, encoding, splits
    metrics  - regression and decision-quality metrics
    models   - linear, RealMLP, TabM, NODE
    optim    - downstream optimization (toy BAP)
    train    - PtO and DFL training loops
    tuning   - Optuna search spaces and runner
"""

__version__ = "0.1.0"

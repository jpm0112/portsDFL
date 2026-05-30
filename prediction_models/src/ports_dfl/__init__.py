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

# This file makes `ports_dfl` an importable Python package. Code that runs on
# `import ports_dfl` goes here; right now it just declares the version string.
# `__version__` is a conventional dunder (double-underscore) name tools read to
# learn the package version.
__version__ = "0.1.0"

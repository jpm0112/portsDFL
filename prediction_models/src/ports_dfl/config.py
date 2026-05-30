"""Project-wide configuration: paths, seeds, CV settings.

Single source of truth for path resolution and reproducibility settings.
Imported by data loaders, training scripts, and test harnesses.
"""

import random
from pathlib import Path  # Path = an object-oriented way to handle file/folder paths

import numpy as np
import torch

# --- Paths ------------------------------------------------------------------
# Resolve the project root from this file's location:
#   src/ports_dfl/config.py  ->  prediction_models/
# `NAME: Path = ...` is a type hint: it tells readers/tools NAME holds a Path.
# `__file__` = the path to THIS .py file. `.resolve()` makes it absolute (no
# "..") and `.parents[2]` walks up 2 folders (parents[0] is the folder holding
# this file, parents[1] the next one up, parents[2] two levels up).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# The `/` operator is overloaded by Path to join paths (like os.path.join).
DATA_PATH: Path = PROJECT_ROOT / "training_dataset.csv"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
OPTUNA_DB_DIR: Path = PROJECT_ROOT / "optuna_studies"

# --- Reproducibility --------------------------------------------------------
# A fixed "seed" makes random number generators produce the same sequence
# every run, so experiments are repeatable.
SEED: int = 42

# --- Cross-validation -------------------------------------------------------
# Split the data into 5 folds: train on 4, validate on 1, rotate, repeat.
N_FOLDS: int = 5
# Stratify by this column so each fold keeps the same proportion of every
# site — prevents a fold from accidentally missing a site entirely.
CV_STRATIFY_COL: str = "Sitio"  # 9 levels — keeps each site in every fold

# --- Target -----------------------------------------------------------------
# The column the models try to predict (the "y" / label).
TARGET_COL: str = "service_time_hours"

# --- Feature roles (for ColumnTransformer) ----------------------------------
# These lists group columns by how they should be preprocessed.
# `list[str]` is a type hint meaning "a list of strings".
# Low-cardinality = few distinct values, so one-hot encoding is cheap/safe.
LOW_CARDINALITY_CATEGORICAL: list[str] = [
    "Sitio",                  # 9 levels
    "Tipo nave (agrupado)",   # 7 levels
    "covid_era",              # 3 levels
]

# High-cardinality = many distinct values; these usually need a different
# encoding (e.g. target/hashing) to avoid an explosion of one-hot columns.
HIGH_CARDINALITY_CATEGORICAL: list[str] = [
    "Puerto origen",
    "Puerto destino",
    "Servicio",
    "Línea naviera",
    "Agencia",
]

# Numeric (continuous) columns. The *_sin/*_cos pairs encode cyclical time
# features (hour, day-of-week, month) so the model sees, e.g., that hour 23
# is close to hour 0 instead of far apart.
NUMERIC_FEATURES: list[str] = [
    "TRG",
    "Calado arribo",
    "Calado diff",
    "atraque_hour_sin",
    "atraque_hour_cos",
    "atraque_dayofweek_sin",
    "atraque_dayofweek_cos",
    "atraque_month_sin",
    "atraque_month_cos",
]

# `+` on lists concatenates them, so ALL_FEATURES is every feature column
# in one list. The parentheses just allow the expression to span two lines.
ALL_FEATURES: list[str] = (
    LOW_CARDINALITY_CATEGORICAL + HIGH_CARDINALITY_CATEGORICAL + NUMERIC_FEATURES
)

# --- Device -----------------------------------------------------------------
# Pick the GPU ("cuda") if PyTorch can see one, otherwise fall back to the CPU.
# The `A if COND else B` is a ternary (inline if): evaluates to A when COND is
# True, else B. Tensors/models can later be moved onto DEVICE with .to(DEVICE).
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# `def name(arg: int) -> None:` defines a function. `seed: int` is a typed
# parameter; `-> None` means the function returns nothing (it just has side
# effects). The triple-quoted string right under `def` is a docstring (help).
def set_seed(seed: int) -> None:
    """Reseed Python, NumPy, and PyTorch for reproducibility.

    Call this at the start of every training entry point. Does not enable
    deterministic CUDA kernels (that costs throughput); use this for
    seedable everyday reproducibility, not for bit-exact GPU determinism.

    Args:
        seed: integer seed to apply to all RNG streams.
    """
    random.seed(seed)        # seed Python's built-in `random` module
    np.random.seed(seed)     # seed NumPy's global random state
    torch.manual_seed(seed)  # seed PyTorch's CPU random generator
    # Only seed the GPU generator(s) when a CUDA device actually exists,
    # otherwise this call would be wasted (and could error on CPU-only setups).
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # cover all GPUs if there are several

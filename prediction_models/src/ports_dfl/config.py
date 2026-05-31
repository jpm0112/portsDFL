"""Project-wide configuration: paths, seeds, CV settings.

Single source of truth for path resolution and reproducibility settings.
Imported by data loaders, training scripts, and test harnesses.
"""

import random
from pathlib import Path

import numpy as np
import torch

# --- Paths ------------------------------------------------------------------
# Resolve project root from this file's location:
#   src/ports_dfl/config.py  ->  prediction_models/
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_PATH: Path = PROJECT_ROOT / "training_dataset.csv"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
OPTUNA_DB_DIR: Path = PROJECT_ROOT / "optuna_studies"

# --- Reproducibility --------------------------------------------------------
SEED: int = 42

# --- Cross-validation -------------------------------------------------------
N_FOLDS: int = 5
# Stratify so each fold keeps the same proportion of every site — prevents a
# fold from accidentally missing a site entirely.
CV_STRATIFY_COL: str = "Sitio"  # 9 levels — keeps each site in every fold

# --- Target -----------------------------------------------------------------
TARGET_COL: str = "service_time_hours"

# --- Feature roles (for ColumnTransformer) ----------------------------------
# Low-cardinality = few distinct values, so one-hot encoding is cheap/safe.
LOW_CARDINALITY_CATEGORICAL: list[str] = [
    "Sitio",                  # 9 levels
    "Tipo nave (agrupado)",   # 7 levels
    "covid_era",              # 3 levels
]

# High-cardinality = many distinct values; need a different encoding
# (e.g. target/hashing) to avoid an explosion of one-hot columns.
HIGH_CARDINALITY_CATEGORICAL: list[str] = [
    "Puerto origen",
    "Puerto destino",
    "Servicio",
    "Línea naviera",
    "Agencia",
]

# The *_sin/*_cos pairs encode cyclical time features (hour, day-of-week,
# month) so the model sees, e.g., that hour 23 is close to hour 0.
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

ALL_FEATURES: list[str] = (
    LOW_CARDINALITY_CATEGORICAL + HIGH_CARDINALITY_CATEGORICAL + NUMERIC_FEATURES
)

# --- Device -----------------------------------------------------------------
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Reseed Python, NumPy, and PyTorch for reproducibility.

    Call this at the start of every training entry point. Does not enable
    deterministic CUDA kernels (that costs throughput); use this for
    seedable everyday reproducibility, not for bit-exact GPU determinism.

    Args:
        seed: integer seed to apply to all RNG streams.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # cover all GPUs if there are several

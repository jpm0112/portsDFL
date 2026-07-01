"""Model feature columns, copied verbatim from ports_dfl.config.

The portable predictor needs this list to feed the fitted preprocessor, but must
avoid importing ports_dfl.config (which pulls in PyTorch). Keep in sync with
config.py if the training feature set ever changes.
"""

from __future__ import annotations

LOW_CARDINALITY_CATEGORICAL: list[str] = [
    "Sitio",
    "Tipo nave (agrupado)",
    "covid_era",
]

HIGH_CARDINALITY_CATEGORICAL: list[str] = [
    "Puerto origen",
    "Puerto destino",
    "Servicio",
    "Línea naviera",
    "Agencia",
]

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

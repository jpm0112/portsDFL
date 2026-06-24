"""Engineer the 17 model features from a RAW vessel CSV (best-effort).

Reproduces the model's feature columns from raw vessel fields. Most transforms are
exact: vessel-type grouping is copied verbatim from
``data_pipeline/build_training_dataset.py``; ``Calado arribo`` is the mean arrival
draft; the six ``atraque_*`` cyclical encodings were verified against the training
data; ``Agencia`` is kept RAW (``"RUT - NAME"``) exactly as the models saw it.

Two features are approximate (see ../predictor/README.md):
  - ``covid_era`` cutoff dates are reverse-engineered (negligible; recent vessels = "post").
  - ``Calado diff`` = arrival - departure draft, so it needs the DEPARTURE drafts
    (known only after the call). Omit them and it defaults to 0.

Rare categorical values that the training pipeline frequency-bucketed
(``other_destinations`` / ``other_liners``) cannot be reproduced here, so an
out-of-vocabulary value falls back to the model's prior at predict time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ports_dfl.config import ALL_FEATURES

# Verbatim from data_pipeline/build_training_dataset.py (VESSEL_TYPE_GROUP).
VESSEL_TYPE_GROUP = {
    "Contenedor": "Container",
    "Carga Seca Granel": "Dry Bulk",
    "Mineral/Granel/Petrolero": "Dry Bulk",
    "Autero": "Vehicle Carrier",
    "Autotrasbordo": "Vehicle Carrier",
    "Transporte Quimico": "Liquid Bulk",
    "Transporte Liquido": "Liquid Bulk",
    "Transporte de Asfalto": "Liquid Bulk",
    "Petrolero": "Liquid Bulk",
    "Tradicional": "General Cargo",
    "Carga de Proyecto": "General Cargo",
    "Chipero": "General Cargo",
    "Refrigerado": "General Cargo",
    "Otros": "General Cargo",
    "Pasajeros": "Passenger",
    "Nave Armada": "Other",
}

# COVID-era boundaries (WHO pandemic declaration / end of the public-health emergency).
# Approximate -- the exact cutoffs aren't in the committed pipeline.
COVID_START = pd.Timestamp("2020-03-11")
COVID_END = pd.Timestamp("2023-05-05")

# Raw columns the user must supply. Departure drafts (for Calado diff) are optional.
REQUIRED_COLUMNS = [
    "Sitio",
    "Tipo nave",
    "arrival_datetime",
    "berthing_datetime",
    "Puerto origen",
    "Puerto destino",
    "Agencia",
    "Línea naviera",
    "Servicio",
    "TRG",
    "draft_arrival_bow",
    "draft_arrival_stern",
]


def _cyclical(values: pd.Series, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (sin, cos) of ``2*pi*values/period`` -- the model's cyclical encoding."""
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a raw vessel DataFrame to the 17 model feature columns.

    Args:
        raw: DataFrame with at least ``REQUIRED_COLUMNS``. Optionally
            ``draft_departure_bow``/``draft_departure_stern`` (for ``Calado diff``) or
            a precomputed ``Calado diff`` column.

    Returns:
        DataFrame with exactly ``config.ALL_FEATURES`` (in order), ready for
        :func:`ports_dfl.inference.predict_frame`.

    Raises:
        ValueError: if a required column is missing or a vessel type is unknown.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw CSV is missing required columns: {missing}")

    out = pd.DataFrame(index=raw.index)
    # Passthrough categoricals. Agencia is kept RAW (matches training); missing
    # line/service use the data's own fill labels.
    out["Sitio"] = raw["Sitio"]
    out["Puerto origen"] = raw["Puerto origen"]
    out["Puerto destino"] = raw["Puerto destino"]
    out["Agencia"] = raw["Agencia"]
    out["Línea naviera"] = raw["Línea naviera"].fillna("NON-LINER")
    out["Servicio"] = raw["Servicio"].fillna("NO SERVICE")
    out["TRG"] = raw["TRG"]

    # Vessel-type grouping: fail loud on an unknown type, like training.
    unmapped = set(raw["Tipo nave"].dropna().unique()) - set(VESSEL_TYPE_GROUP)
    if unmapped:
        raise ValueError(f"Unknown vessel type(s) not in VESSEL_TYPE_GROUP: {sorted(unmapped)}")
    out["Tipo nave (agrupado)"] = raw["Tipo nave"].map(VESSEL_TYPE_GROUP)

    # Drafts.
    arrival_mean = (raw["draft_arrival_bow"] + raw["draft_arrival_stern"]) / 2
    out["Calado arribo"] = arrival_mean
    if "Calado diff" in raw.columns:
        out["Calado diff"] = raw["Calado diff"]
    elif {"draft_departure_bow", "draft_departure_stern"} <= set(raw.columns):
        departure_mean = (raw["draft_departure_bow"] + raw["draft_departure_stern"]) / 2
        out["Calado diff"] = arrival_mean - departure_mean
    else:
        out["Calado diff"] = 0.0  # unknown before departure -- see README caveat

    # covid_era from the arrival date.
    arrival = pd.to_datetime(raw["arrival_datetime"])
    out["covid_era"] = np.select(
        [arrival < COVID_START, arrival < COVID_END], ["pre", "during"], default="post"
    )

    # Cyclical encodings from the berthing (atraque / first-mooring) datetime.
    berthing = pd.to_datetime(raw["berthing_datetime"])
    out["atraque_hour_sin"], out["atraque_hour_cos"] = _cyclical(berthing.dt.hour, 24)
    out["atraque_dayofweek_sin"], out["atraque_dayofweek_cos"] = _cyclical(berthing.dt.dayofweek, 7)
    out["atraque_month_sin"], out["atraque_month_cos"] = _cyclical(berthing.dt.month, 12)

    return out[list(ALL_FEATURES)]

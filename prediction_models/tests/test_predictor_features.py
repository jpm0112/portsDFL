"""Tests for the raw->feature engineering in predictor/features.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# The predictor tool lives in a sibling subdir (not a package); add it to the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "predictor"))
from features import engineer, unseen_values  # noqa: E402

from ports_dfl.config import ALL_FEATURES  # noqa: E402


def _raw_row(**overrides):
    """A single valid raw vessel row, with optional field overrides."""
    base = {
        "Sitio": "Sitio 1",
        "Tipo nave": "Contenedor",
        "arrival_datetime": "2024-06-15 08:30",
        "berthing_datetime": "2024-06-15 14:00",
        "Puerto origen": "LIRQUEN",
        "Puerto destino": "ANGAMOS",
        "Agencia": "96707720-8 - MSC",
        "Línea naviera": "MSC",
        "Servicio": "Andes",
        "TRG": 141635,
        "draft_arrival_bow": 12.5,
        "draft_arrival_stern": 12.9,
        "draft_departure_bow": 11.8,
        "draft_departure_stern": 12.2,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_engineer_produces_exact_feature_schema() -> None:
    out = engineer(_raw_row())
    assert list(out.columns) == list(ALL_FEATURES)
    assert len(out) == 1


def test_draft_features() -> None:
    out = engineer(_raw_row())
    assert out["Calado arribo"].iloc[0] == pytest.approx((12.5 + 12.9) / 2)
    assert out["Calado diff"].iloc[0] == pytest.approx((12.5 + 12.9) / 2 - (11.8 + 12.2) / 2)


def test_calado_diff_defaults_to_zero_without_departure_drafts() -> None:
    raw = _raw_row().drop(columns=["draft_departure_bow", "draft_departure_stern"])
    assert engineer(raw)["Calado diff"].iloc[0] == 0.0


def test_vessel_type_grouping() -> None:
    assert engineer(_raw_row())["Tipo nave (agrupado)"].iloc[0] == "Container"
    assert engineer(_raw_row(**{"Tipo nave": "Pasajeros"}))["Tipo nave (agrupado)"].iloc[0] == "Passenger"


def test_unknown_vessel_type_raises() -> None:
    with pytest.raises(ValueError, match="vessel type"):
        engineer(_raw_row(**{"Tipo nave": "NOT A REAL TYPE"}))


def test_missing_required_column_raises() -> None:
    with pytest.raises(ValueError, match="TRG"):
        engineer(_raw_row().drop(columns=["TRG"]))


def test_covid_era_cutoffs() -> None:
    assert engineer(_raw_row(arrival_datetime="2019-01-01"))["covid_era"].iloc[0] == "pre"
    assert engineer(_raw_row(arrival_datetime="2021-06-01"))["covid_era"].iloc[0] == "during"
    assert engineer(_raw_row(arrival_datetime="2024-06-15"))["covid_era"].iloc[0] == "post"


def test_missing_categoricals_filled() -> None:
    out = engineer(_raw_row(**{"Línea naviera": np.nan, "Servicio": np.nan}))
    assert out["Línea naviera"].iloc[0] == "NON-LINER"
    assert out["Servicio"].iloc[0] == "NO SERVICE"


def test_cyclical_encoding_matches_formula() -> None:
    # berthing at 14:00 -> hour 14 -> sin/cos(2*pi*14/24).
    out = engineer(_raw_row(berthing_datetime="2024-06-15 14:00"))
    assert out["atraque_hour_sin"].iloc[0] == pytest.approx(np.sin(2 * np.pi * 14 / 24))
    assert out["atraque_hour_cos"].iloc[0] == pytest.approx(np.cos(2 * np.pi * 14 / 24))


def test_blank_datetime_raises() -> None:
    with pytest.raises(ValueError, match="datetime"):
        engineer(_raw_row(berthing_datetime=""))
    with pytest.raises(ValueError, match="datetime"):
        engineer(_raw_row(arrival_datetime=np.nan))


def test_unseen_values_flags_only_unknowns() -> None:
    feats = engineer(_raw_row(**{"Puerto origen": "MADE_UP_PORT"}))
    vocab = {"Puerto origen": ["LIRQUEN", "CALLAO"], "Sitio": ["Sitio 1", "Sitio 2"]}
    # Sitio 1 is in the vocab (not flagged); the made-up port is not.
    assert unseen_values(feats, vocab) == {"Puerto origen": ["MADE_UP_PORT"]}

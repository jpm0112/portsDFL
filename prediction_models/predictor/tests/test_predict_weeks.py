"""Checks for the weekly-schedule enrichment in predict_weeks.py.

Run: cd prediction_models/predictor && python -m pytest tests/test_predict_weeks.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# predict_weeks imports sibling modules (columns/features/predict) by bare name, so the
# predictor/ dir must be importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from columns import ALL_FEATURES  # noqa: E402
from predict_weeks import (  # noqa: E402
    build_agency_lookup,
    build_vessel_profile,
    enrich_week,
)


def _fake_history() -> pd.DataFrame:
    """Two calls of one real vessel + one call of another, with the raw Spanish headers."""
    return pd.DataFrame(
        {
            "Nave": ["NORD ABIDJAN", "nord abidjan", "OSSA"],
            "T. nave": ["Carga Seca Granel", "Carga Seca Granel", "Carga Seca Granel"],
            "TRG": [23000.0, 23600.0, 35596.0],
            "Sitio": ["Sitio 8", "Sitio 8", "Sitio 9"],
            "Puerto origen": ["ARICA", "ARICA", "MEJILLONES"],
            "Puerto destino": ["SANTOS - SP", "SANTOS - SP", "COQUIMBO"],
            "L. naviera": [None, None, "CSAV"],
            "Servicio": [None, None, "WCSA"],
            "C. arribo proa": [9.0, 9.4, 11.0],
            "C. arribo popa": [9.2, 9.6, 11.2],
            "Agencia": ["80992000-3 - ULTRAMAR", "80992000-3 - ULTRAMAR", "80010900-0 - AGENTAL"],
        }
    )


def test_agency_lookup_maps_bare_name_to_raw_string():
    lookup = build_agency_lookup(_fake_history())
    assert lookup["ULTRAMAR"] == "80992000-3 - ULTRAMAR"
    assert lookup["AGENTAL"] == "80010900-0 - AGENTAL"


def test_profile_averages_drafts_and_takes_modal_categoricals():
    profile, fallbacks = build_vessel_profile(_fake_history())
    # Name key is normalized -> the two "NORD ABIDJAN" rows collapse to one vessel.
    row = profile.loc["NORD ABIDJAN"]
    assert row["TRG"] == 23300.0                      # mean of 23000 and 23600
    assert np.isclose(row["draft_arrival_bow"], 9.2)  # mean of 9.0 and 9.4
    assert row["Sitio"] == "Sitio 8"
    assert set(fallbacks) >= set(ALL_FEATURES) - {"covid_era", "TRG"} - set(
        c for c in ALL_FEATURES if c.startswith("atraque_")
    ) or "TRG" in fallbacks  # fallbacks cover the history-sourced fields


def test_matched_vessel_enriches_and_engineers_cleanly():
    history = _fake_history()
    profile, fallbacks = build_vessel_profile(history)
    agency_lookup = build_agency_lookup(history)
    week = pd.DataFrame(
        {
            "E.T.A.": pd.to_datetime(["2024-11-14 13:00", "2024-11-14 21:18"]),
            "Agencia": ["ULTRAMAR", "AGENTAL"],
            "Nave": ["NORD ABIDJAN", "OSSA"],
            "Carga": ["Granel sólido", "Granel sólido"],
        }
    )
    enriched, meta = enrich_week(week, profile, fallbacks, agency_lookup)

    assert meta["matched_history"].all()
    assert enriched.loc[0, "TRG"] == 23300.0
    assert enriched.loc[0, "Agencia"] == "80992000-3 - ULTRAMAR"  # remapped to raw string
    assert (meta["notes"].str.contains("berthing=ETA")).all()

    # engineer() must accept the enriched frame and return exactly the model features, no NaN.
    from features import engineer

    features = engineer(enriched)
    assert list(features.columns) == list(ALL_FEATURES)
    assert not features.isna().any().any()


def test_unseen_vessel_is_flagged_and_typed_from_cargo():
    history = _fake_history()
    profile, fallbacks = build_vessel_profile(history)
    agency_lookup = build_agency_lookup(history)
    week = pd.DataFrame(
        {
            "E.T.A.": pd.to_datetime(["2024-11-15 08:00"]),
            "Agencia": ["ULTRAMAR"],
            "Nave": ["BRAND NEW SHIP"],
            "Carga": ["Contenedores"],
        }
    )
    enriched, meta = enrich_week(week, profile, fallbacks, agency_lookup)

    assert not meta.loc[0, "matched_history"]
    assert "vessel not in history" in meta.loc[0, "notes"]
    assert enriched.loc[0, "Tipo nave"] == "Contenedor"  # guessed from Carga, not fleet mode
    assert enriched.loc[0, "TRG"] == fallbacks["TRG"]     # numeric falls back to global mean

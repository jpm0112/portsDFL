"""Batch-predict berth service time for weekly vessel-schedule files.

An operational weekly schedule (e.g. ``weeks/semana1.xlsx``) only carries a handful of
fields -- ``E.T.A., Agencia, Nave, Eslora, Terminal, Emp. muellaje, Carga, Detalle`` --
none of the heavy predictors the tree models need (TRG, vessel type, drafts, ports, Sitio,
line, service). This script recovers those by joining each vessel **by name** to its
history in ``data/BBDD limpia(1).xlsx`` (per-vessel mode for categoricals, mean for
numerics), engineers the 17 model features via :func:`features.engineer`, runs the tree
models (rf/xgb/lgbm), and writes one predictions CSV per weekly file.

    python predict_weeks.py                              # weeks/ -> predictions/
    python predict_weeks.py --weeks-dir weeks --out-dir predictions

Known approximation: the weekly files have no actual berthing time, so ``E.T.A.`` is used
for both arrival and berthing. That mis-times the 6 cyclical berthing features somewhat;
every output row carries a ``berthing≈ETA`` note. A vessel absent from history is still
predicted (type guessed from ``Carga``, numerics from global means) and flagged
``matched_history=False``.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from columns import ALL_FEATURES
from features import REQUIRED_COLUMNS, VESSEL_TYPE_GROUP, engineer, unseen_values
from predict import ARTIFACTS, BEST_MODEL, TREE_MODELS, load_estimator

HERE = Path(__file__).resolve().parent
DEFAULT_WEEKS_DIR = HERE / "weeks"
DEFAULT_OUT_DIR = HERE / "predictions"
DEFAULT_HISTORY = HERE.parent.parent / "data" / "BBDD limpia(1).xlsx"
HISTORY_SHEET = "Resume Naves Comerciales (4)"

# Raw history header (Spanish, per data_pipeline/build_training_dataset.py COLUMN_MAP) ->
# the REQUIRED_COLUMNS name we fill. Categoricals take the per-vessel mode, numerics the mean.
CATEGORICAL_FROM_HISTORY = {
    "T. nave": "Tipo nave",
    "Sitio": "Sitio",
    "Puerto origen": "Puerto origen",
    "Puerto destino": "Puerto destino",
    "L. naviera": "Línea naviera",
    "Servicio": "Servicio",
}
NUMERIC_FROM_HISTORY = {
    "TRG": "TRG",
    "C. arribo proa": "draft_arrival_bow",
    "C. arribo popa": "draft_arrival_stern",
}

# For a vessel absent from history, guess its raw vessel type from the weekly "Carga" field.
# Values must be VESSEL_TYPE_GROUP keys so engineer() won't raise; unmatched -> "Otros".
CARGA_TO_TYPE = {
    "contenedor": "Contenedor",
    "granel solido": "Carga Seca Granel",
    "granel liquido": "Transporte Liquido",
    "vehiculo": "Autero",
    "pasajero": "Pasajeros",
}


def _norm_name(value: object) -> str:
    """Uppercase, strip, and collapse internal whitespace -- the vessel-name join key."""
    return " ".join(str(value).upper().split())


def _strip_accents(text: str) -> str:
    """Drop diacritics so 'Granel sólido' matches the ASCII keys in CARGA_TO_TYPE."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _first_mode(series: pd.Series) -> object:
    """Most common non-null value (first if tied); NaN if the series is all-null."""
    non_null = series.dropna()
    return non_null.mode().iloc[0] if not non_null.empty else np.nan


def _carga_to_type(carga: object) -> str:
    """Map a weekly 'Carga' label to a raw vessel type (a VESSEL_TYPE_GROUP key)."""
    key = _strip_accents(str(carga)).lower()
    for needle, vessel_type in CARGA_TO_TYPE.items():
        if needle in key:
            return vessel_type
    return "Otros"  # -> General Cargo


def build_vessel_profile(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Per-vessel attribute profile plus global fallbacks, keyed on the normalized name.

    Args:
        history: raw BBDD-limpia rows (original Spanish headers).

    Returns:
        (profile, fallbacks): ``profile`` indexed by ``_norm_name(Nave)`` with one column per
        REQUIRED_COLUMNS field sourced from history (mode/mean); ``fallbacks`` the same fields
        computed over the whole fleet, for vessels absent from history.
    """
    keyed = history.assign(_key=history["Nave"].map(_norm_name))
    named_aggs = {
        out: pd.NamedAgg(column=raw, aggfunc=_first_mode)
        for raw, out in CATEGORICAL_FROM_HISTORY.items()
    }
    named_aggs.update(
        {out: pd.NamedAgg(column=raw, aggfunc="mean") for raw, out in NUMERIC_FROM_HISTORY.items()}
    )
    profile = keyed.groupby("_key").agg(**named_aggs)

    fallbacks: dict[str, object] = {
        out: _first_mode(history[raw]) for raw, out in CATEGORICAL_FROM_HISTORY.items()
    }
    fallbacks.update({out: history[raw].mean() for raw, out in NUMERIC_FROM_HISTORY.items()})
    return profile, fallbacks


def build_agency_lookup(history: pd.DataFrame) -> dict[str, str]:
    """Map an uppercased agency NAME to the raw ``"RUT - NAME"`` string the model was trained on.

    The weekly files give a bare name ("ULTRAMAR"); the model's Agencia vocab is the full
    ``"80992000-3 - ULTRAMAR"``. Without this remap every agency is out-of-vocabulary.
    """
    raw = history["Agencia"].dropna().astype(str)
    dashed = raw[raw.str.contains(" - ")]
    names = dashed.str.split(" - ", n=1).str[1].str.strip().str.upper()
    pairs = pd.DataFrame({"name": names.to_numpy(), "raw": dashed.to_numpy()})
    return {name: group.mode().iloc[0] for name, group in pairs.groupby("name")["raw"]}


def enrich_week(
    week: pd.DataFrame,
    profile: pd.DataFrame,
    fallbacks: dict[str, object],
    agency_lookup: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backfill a weekly schedule into the 12 REQUIRED_COLUMNS the model needs.

    Returns:
        (enriched, meta): ``enriched`` has exactly REQUIRED_COLUMNS; ``meta`` has
        ``matched_history`` (bool) and ``notes`` (imputation caveats), aligned to ``week``.
    """
    keys = week["Nave"].map(_norm_name)
    matched = keys.isin(profile.index)
    aligned = profile.reindex(keys.to_numpy())
    aligned.index = week.index

    enriched = pd.DataFrame(index=week.index)
    for out in list(CATEGORICAL_FROM_HISTORY.values()) + list(NUMERIC_FROM_HISTORY.values()):
        enriched[out] = aligned[out].where(aligned[out].notna(), fallbacks[out])

    # Unseen vessels have no history type -> guess it from the weekly cargo instead of the
    # fleet-wide mode (a container ship shouldn't inherit "bulk" just because bulk is common).
    unseen = ~matched.to_numpy()
    if unseen.any():
        enriched.loc[unseen, "Tipo nave"] = week.loc[unseen, "Carga"].map(_carga_to_type)

    # E.T.A. drives arrival; no real berthing time exists in a schedule, so reuse it (caveat below).
    enriched["arrival_datetime"] = pd.to_datetime(week["E.T.A."], errors="coerce")
    enriched["berthing_datetime"] = enriched["arrival_datetime"]

    agency_names = week["Agencia"].astype(str).str.strip().str.upper()
    enriched["Agencia"] = agency_names.map(agency_lookup)
    agency_unmatched = enriched["Agencia"].isna()
    # Unmatched -> keep the raw name; it falls back to the encoder's prior at predict time.
    enriched.loc[agency_unmatched, "Agencia"] = week.loc[agency_unmatched, "Agencia"]

    notes = pd.Series("berthing=ETA (schedule has no real berthing time)", index=week.index)
    notes = notes.where(matched, notes + "; vessel not in history (imputed)")
    notes = notes.where(~agency_unmatched, notes + "; agency unmatched")
    meta = pd.DataFrame({"matched_history": matched.to_numpy(), "notes": notes.to_numpy()},
                        index=week.index)
    return enriched[REQUIRED_COLUMNS], meta


def predict_week(features: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Transform engineered features and run each tree model; add an ensemble mean."""
    vocab_path = ARTIFACTS / "vocab.json"
    if vocab_path.exists():
        for col, vals in unseen_values(features, json.loads(vocab_path.read_text("utf-8"))).items():
            print(f"  ! {col}: {vals} not seen in training; using that field's overall average")

    preprocessor = joblib.load(ARTIFACTS / "preprocessor.pkl")
    transformed = preprocessor.transform(features[ALL_FEATURES]).astype(np.float32)
    preds = pd.DataFrame(index=features.index)
    for name in models:
        preds[name] = np.clip(np.asarray(load_estimator(name).predict(transformed)).ravel(), 0.0, None)
    preds["ensemble_mean"] = preds.mean(axis=1)
    return preds


def _read_week(path: Path) -> pd.DataFrame:
    """Read a weekly schedule file (.xlsx or .csv)."""
    return pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich weekly vessel schedules from history and predict berth service time (hours)."
    )
    parser.add_argument("--weeks-dir", type=Path, default=DEFAULT_WEEKS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument(
        "--models", default=None, help=f"comma-separated subset of {TREE_MODELS} (default: all)"
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else list(TREE_MODELS)
    unknown = [m for m in models if m not in TREE_MODELS]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}. Available: {TREE_MODELS}")

    week_files = sorted(
        p for p in args.weeks_dir.iterdir() if p.suffix.lower() in {".xlsx", ".csv"}
    )
    if not week_files:
        raise SystemExit(f"No .xlsx/.csv files found in {args.weeks_dir}")

    history = pd.read_excel(args.history, sheet_name=HISTORY_SHEET, engine="openpyxl")
    profile, fallbacks = build_vessel_profile(history)
    agency_lookup = build_agency_lookup(history)
    print(f"Loaded history: {len(history)} calls, {len(profile)} vessels, "
          f"{len(agency_lookup)} agencies. Best model = {BEST_MODEL}.\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in week_files:
        week = _read_week(path)
        enriched, meta = enrich_week(week, profile, fallbacks, agency_lookup)
        features = engineer(enriched)
        preds = predict_week(features, models)

        result = pd.concat(
            [
                week.reset_index(drop=True),
                enriched[["Tipo nave", "TRG", "Sitio"]].reset_index(drop=True),
                preds.reset_index(drop=True),
                meta.reset_index(drop=True),
            ],
            axis=1,
        )
        out_path = args.out_dir / f"{path.stem}_predictions.csv"
        result.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  {path.name}: {len(result)} vessels -> {out_path.name} "
              f"({int(meta['matched_history'].sum())}/{len(meta)} matched history, "
              f"{BEST_MODEL} mean {preds[BEST_MODEL].mean():.1f} h)")


if __name__ == "__main__":
    main()

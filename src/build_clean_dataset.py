"""
Build a clean, source-faithful CSV from BBDD limpia.

Reads the full vessel-call sheet from the original Excel workbook, drops
the 8 known anomalies (berth stay < 2h or > 500h), expands abbreviated
column names to their full Spanish form, removes columns that carry no
extra information (`Año`, `Mes`, `Nave`), and adds an aggregated
`Tipo nave (agrupado)` column that maps the 16 raw vessel types into the
7 operational groups documented in `data/column_description.pdf`.

Each transformation is implemented as its own function so the steps can be
audited, reordered, or unit-tested independently. The output is the
canonical "clean" dataset that complements the engineered
`training_dataset.csv`.

Input:  data/BBDD limpia(1).xlsx -> sheet "Resume Naves Comerciales (4)"
Output: data/clean_dataset.csv   (5,597 rows x 28 columns, UTF-8)
"""

import os
import pandas as pd

# --- Configuration -------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
INPUT_FILE = os.path.join(DATA_DIR, "BBDD limpia(1).xlsx")
SHEET_NAME = "Resume Naves Comerciales (4)"
OUTPUT_FILE = os.path.join(DATA_DIR, "clean_dataset.csv")

# Berth-stay thresholds (hours), mirroring build_training_dataset.py.
# Below MIN -> aborted call, above MAX -> single 780h outlier.
MIN_BERTH_HOURS = 2
MAX_BERTH_HOURS = 500

# Abbreviated -> full Spanish column names. The original BBDD sheet uses
# period-style shorthands (e.g. "L. naviera") that are ambiguous to anyone
# not familiar with the source. Expanding them once here keeps every
# downstream artefact self-explanatory.
ABBREVIATION_MAP = {
    "Cód. nave": "Código nave",
    "T. nave": "Tipo nave",
    "C. arribo proa": "Calado arribo proa",
    "C. arribo popa": "Calado arribo popa",
    "C. zarpe proa": "Calado zarpe proa",
    "C. zarpe popa": "Calado zarpe popa",
    "L. naviera": "Línea naviera",
    "F. arribo": "Fecha arribo",
}

# `Fecha` (booking/registration date) shares its name with the operational
# date columns once the abbreviations above are expanded. Tag it with
# `(solicitud)` to disambiguate.
FECHA_RENAME = {"Fecha": "Fecha (solicitud)"}

# Columns dropped because they are 100% derivable from other columns or
# carry no information for the modelling task:
#   - `Año`, `Mes` are derived from `Fecha arribo` (verified: 100% match).
#   - `Nave` is the vessel name; `Código nave` (IMO number) is the unique,
#     stable identifier. 28 vessels appear in the data under more than one
#     name (renames during the period), so always group by `Código nave`.
COLUMNS_TO_DROP = ["Año", "Mes", "Nave"]

# EPSA and QC label the same physical liquid-bulk berth (Sitio 9). EPSA
# (Empresa Portuaria San Antonio) operated it directly until mid-2020,
# then a private concessionaire (QC) took over later that year. 11 vessels
# appear in both Terminal labels, confirming it's the same berth. The site
# label is also stored inconsistently ("9" vs "Sitio 9"). Normalise both.
TERMINAL_MERGE = {"EPSA": "QC"}
SITE_NORMALISE = {"9": "Sitio 9"}

# `Línea naviera` and `Servicio` carry two clean-up needs:
#   1. Two HAPAG-LLOYD legal entities ('HAPAG LLOYD CHILE LIMITADA …' n=8
#      vs 'HAPAG-LLOYD CHILE SPA' n=467) refer to the same operator. Merge
#      to the most common name.
#   2. The auto-carrier service is recorded with two casings — 'CAR
#      CARRIERS' (n=294) and 'Car Carriers' (n=113). Same service. Merge
#      to the most common form.
# Missing values in both columns are NOT data errors: they encode tramp /
# non-liner traffic (≥98% of dry bulk, 100% of cruise, 60% of liquid bulk).
# Fill with the literal label NON_LINER_LABEL so the semantic meaning is
# preserved for downstream models.
LINEA_NAME_MAP = {
    "HAPAG LLOYD CHILE LIMITADA P/C DE HAPAG LLOYD AG": "HAPAG-LLOYD CHILE SPA",
}
SERVICIO_NAME_MAP = {
    "Car Carriers": "CAR CARRIERS",
    "CAR CARRIER": "CAR CARRIERS",  # singular variant (81 rows)
}
NON_LINER_LABEL = "NON-LINER"   # used for Línea naviera missingness
NO_SERVICE_LABEL = "NO SERVICE"  # used for Servicio missingness (different label
                                 # so the two columns don't share a value)

# 16 raw `Tipo nave` values -> 7 operational groups. Mapping matches
# `data/column_description.pdf` section 7. Note that "Petrolero" appears
# in two raw labels: `Mineral/Granel/Petrolero` is an OBO (Ore-Bulk-Oil
# combination) carrier and routes to Dry Bulk because in this dataset it
# behaves like a bulker (45h median berth time); pure `Petrolero` is a
# dedicated tanker and routes to Liquid Bulk.
VESSEL_TYPE_GROUPS = {
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


# --- Pipeline steps ------------------------------------------------------

def load_source():
    """
    Load the full BBDD limpia sheet without any modification.

    Output: DataFrame with all 31 original Spanish columns, 5,605 rows.
    """
    print("Loading data from Excel...")
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, engine="openpyxl")
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def filter_anomalies(df):
    """
    Drop rows whose `Estadía sitio` is below MIN_BERTH_HOURS or above
    MAX_BERTH_HOURS. The bounds are conservative and only remove records
    that are unambiguously bad (aborted calls and a single 780h outlier).

    Input:  DataFrame with `Estadía sitio` as Timedelta.
    Output: Filtered DataFrame, same columns as the input.
    """
    n_before = len(df)

    estadia_hours = df["Estadía sitio"].dt.total_seconds() / 3600
    mask_short = estadia_hours < MIN_BERTH_HOURS
    mask_extreme = estadia_hours > MAX_BERTH_HOURS

    n_short = int(mask_short.sum())
    n_extreme = int(mask_extreme.sum())

    df = df.loc[~(mask_short | mask_extreme)].copy()

    print(f"  Removed {n_short} records with berth stay < {MIN_BERTH_HOURS}h")
    print(f"  Removed {n_extreme} records with berth stay > {MAX_BERTH_HOURS}h")
    print(f"  Rows: {n_before} -> {len(df)}")
    return df


def filter_unmoor_after_departure(df):
    """
    Drop rows where `Última espía desatraque > Zarpe`. The vessel cannot
    depart before its last mooring line is released, so these records
    contain a typo in one of the two timestamps. Five of the seven affected
    rows are exactly ~24h off, suggesting a day mistyped during entry.
    Both timestamps feed service-time targets, so leaving the rows in would
    bias `estadia_sitio_hours` upward.

    Input:  DataFrame with `Última espía desatraque` and `Zarpe` as
            datetime-like.
    Output: DataFrame with the bad rows removed.
    """
    n_before = len(df)
    bad = df["Última espía desatraque"] > df["Zarpe"]
    n_bad = int(bad.sum())
    df = df.loc[~bad].copy()
    print(f"  Removed {n_bad} rows where 'Última espía desatraque' > 'Zarpe'")
    print(f"  Rows: {n_before} -> {len(df)}")
    return df


def expand_abbreviations(df):
    """
    Rename period-style abbreviated columns to their full Spanish form
    (e.g. `L. naviera` -> `Línea naviera`). Asserts every abbreviated key
    is present in the source so the script fails loudly if the upstream
    sheet schema changes.

    Input:  DataFrame with abbreviated column names.
    Output: DataFrame with the 8 columns in ABBREVIATION_MAP renamed.
    """
    missing = [k for k in ABBREVIATION_MAP if k not in df.columns]
    if missing:
        raise KeyError(f"Source sheet is missing expected columns: {missing}")
    print(f"  Renaming {len(ABBREVIATION_MAP)} abbreviated columns")
    return df.rename(columns=ABBREVIATION_MAP)


def rename_fecha_to_solicitud(df):
    """
    Disambiguate the booking-date column by renaming `Fecha` to
    `Fecha (solicitud)`. Must run after `expand_abbreviations` so that
    the original `F. arribo` has already become `Fecha arribo` and only
    one column literally named `Fecha` remains.

    Input:  DataFrame with a single column literally named `Fecha`.
    Output: DataFrame with that column renamed to `Fecha (solicitud)`.
    """
    if list(df.columns).count("Fecha") != 1:
        raise ValueError("Expected exactly one column named 'Fecha'.")
    print("  Renaming 'Fecha' -> 'Fecha (solicitud)'")
    return df.rename(columns=FECHA_RENAME)


def drop_redundant_columns(df):
    """
    Drop columns that contain no information not already present elsewhere:
    `Año`, `Mes` (derivable from `Fecha arribo`) and `Nave` (vessel name,
    redundant with the IMO `Código nave`).

    Input:  DataFrame containing every name in COLUMNS_TO_DROP.
    Output: DataFrame without those columns.
    """
    missing = [c for c in COLUMNS_TO_DROP if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot drop missing columns: {missing}")
    print(f"  Dropping {len(COLUMNS_TO_DROP)} redundant columns: {COLUMNS_TO_DROP}")
    return df.drop(columns=COLUMNS_TO_DROP)


def normalise_terminal_and_site(df):
    """
    Merge `EPSA` into `QC` (same physical berth, different operator across
    time) and normalise the inconsistent `Sitio` label "9" -> "Sitio 9".
    Verifies that the only "9" values belong to the (formerly EPSA, now QC)
    rows so we don't accidentally rename an unrelated record.

    Input:  DataFrame with `Terminal` and `Sitio` columns.
    Output: DataFrame with merged terminal and consistent site labels.
    """
    n_epsa = (df["Terminal"] == "EPSA").sum()
    df = df.copy()
    df["Terminal"] = df["Terminal"].replace(TERMINAL_MERGE)

    # Cast Sitio to string first: BBDD stores the EPSA-era value as integer 9
    # (not the string "9"), so a plain `replace` against the string key would
    # silently miss those 23 rows. Stringifying makes the comparison robust to
    # the source dtype.
    df["Sitio"] = df["Sitio"].astype(str)
    bad_sites = df.loc[df["Sitio"] == "9"]
    if not bad_sites["Terminal"].eq("QC").all():
        raise ValueError("Found 'Sitio = 9' outside the QC terminal — refusing to rename.")
    df["Sitio"] = df["Sitio"].replace(SITE_NORMALISE)

    print(f"  Merged {n_epsa} EPSA rows into QC")
    print(f"  Normalised Sitio label '9' -> 'Sitio 9'")
    return df


def normalise_linea_and_servicio(df):
    """
    Consolidate duplicate naming and fill structural missingness in the
    `Línea naviera` and `Servicio` columns. The 30% missing rate in both
    columns reflects vessels that are not on a scheduled liner service
    (tramp / spot vessels), so the missing values are filled with the
    explicit label NON_LINER_LABEL rather than an opaque "UNKNOWN".

    Input:  DataFrame with `Línea naviera` and `Servicio` columns.
    Output: DataFrame with name variants merged and NaNs replaced.
    """
    n_lin_dup  = df["Línea naviera"].isin(LINEA_NAME_MAP).sum()
    n_serv_dup = df["Servicio"].isin(SERVICIO_NAME_MAP).sum()
    n_lin_na   = df["Línea naviera"].isna().sum()
    n_serv_na  = df["Servicio"].isna().sum()

    df = df.copy()
    df["Línea naviera"] = df["Línea naviera"].replace(LINEA_NAME_MAP).fillna(NON_LINER_LABEL)
    df["Servicio"]      = df["Servicio"].replace(SERVICIO_NAME_MAP).fillna(NO_SERVICE_LABEL)

    print(f"  Merged {n_lin_dup} HAPAG-LLOYD variant rows into the canonical name")
    print(f"  Merged {n_serv_dup} 'Car Carriers' rows into 'CAR CARRIERS'")
    print(f"  Filled {n_lin_na} missing 'Línea naviera' with '{NON_LINER_LABEL}'")
    print(f"  Filled {n_serv_na} missing 'Servicio' with '{NO_SERVICE_LABEL}'")
    return df


def add_vessel_type_group(df):
    """
    Add `Tipo nave (agrupado)` mapping the 16 raw vessel types to 7
    operational groups (Container, Dry Bulk, Vehicle Carrier, Liquid Bulk,
    General Cargo, Passenger, Other). Inserts the new column right after
    `Tipo nave`. Fails if any raw value in the data is missing from the
    mapping (so new vessel types are never silently dropped).

    Input:  DataFrame with `Tipo nave` populated.
    Output: DataFrame with `Tipo nave (agrupado)` placed next to its source.
    """
    unmapped = set(df["Tipo nave"].dropna().unique()) - set(VESSEL_TYPE_GROUPS)
    if unmapped:
        raise ValueError(f"Vessel types missing from VESSEL_TYPE_GROUPS: {unmapped}")

    df = df.copy()
    df["Tipo nave (agrupado)"] = df["Tipo nave"].map(VESSEL_TYPE_GROUPS)

    cols = list(df.columns)
    cols.remove("Tipo nave (agrupado)")
    cols.insert(cols.index("Tipo nave") + 1, "Tipo nave (agrupado)")
    df = df[cols]

    counts = df["Tipo nave (agrupado)"].value_counts()
    print(f"  Added 'Tipo nave (agrupado)' with {len(counts)} groups:")
    for group, n in counts.items():
        print(f"    {group:18s} {n:5d}  ({n/len(df)*100:5.2f}%)")
    return df


def force_consistent_vessel_type_group(df):
    """
    Some vessels appear in the source under more than one raw `Tipo nave`,
    and a few of them cross operational-group boundaries (e.g. one call as
    `Carga Seca Granel` -> Dry Bulk, another as `Contenedor` -> Container).
    For per-vessel grouped features to be stable, force every row of such a
    vessel to share the same `Tipo nave (agrupado)`. Pick the most common
    group across that vessel's rows. Ties are broken by chronological order
    (earliest `Fecha arribo` wins), preserving the vessel's original
    classification when frequencies are equal. The raw `Tipo nave` column
    is intentionally left untouched so the call-by-call source signal is
    still visible.

    Input:  DataFrame with `Tipo nave (agrupado)` populated and
            `Fecha arribo` available for tie-breaking.
    Output: DataFrame with each vessel's group set to a single value.
    """
    df = df.copy()
    vt = df.groupby("Código nave")["Tipo nave (agrupado)"].nunique()
    inconsistent = vt[vt > 1].index.tolist()
    n_changed = 0
    for code in inconsistent:
        sub_idx = df.index[df["Código nave"] == code]
        sorted_groups = df.loc[sub_idx].sort_values("Fecha arribo")["Tipo nave (agrupado)"]
        counts = sorted_groups.value_counts()
        top = counts.iloc[0]
        tied = counts[counts == top].index.tolist()
        if len(tied) == 1:
            chosen = tied[0]
        else:
            # Tie-break by chronological order: first vessel arrival wins.
            chosen = next(v for v in sorted_groups if v in tied)
        rows_to_change = (df["Código nave"] == code) & (df["Tipo nave (agrupado)"] != chosen)
        n_now = int(rows_to_change.sum())
        if n_now > 0:
            df.loc[rows_to_change, "Tipo nave (agrupado)"] = chosen
            n_changed += n_now
    print(f"  Forced consistent group for {len(inconsistent)} vessel(s); {n_changed} rows changed")
    return df


def _conditional_local_gap(clean, end_col, start_col):
    """
    Compute the mean (in seconds) of `end_col - start_col` over non-violating
    rows, grouped by `(Tipo nave (agrupado), Terminal)`. Returns the per-cell
    series and the global mean as a fallback for cells with no observations.
    """
    secs = (clean[end_col] - clean[start_col]).dt.total_seconds()
    grp = secs.groupby([clean["Tipo nave (agrupado)"], clean["Terminal"]]).mean()
    return grp, secs.mean()


def _lookup_gap(grp_means, global_mean, type_grp, terminal):
    """Group-conditional mean with global fallback when the cell is empty."""
    val = grp_means.get((type_grp, terminal))
    if val is None or pd.isna(val):
        return global_mean
    return val


def fix_ordering_violations(df):
    """
    Repair two ordering violations introduced by data-entry typos:

      V1: `Fecha arribo > Fecha práctico atraque` (2 rows)
      V2: `Fecha recepción nave > Fecha despacho nave` (8 rows)

    For each violating row, decide which of the two timestamps is the typo
    by inspecting its consistency with the surrounding event chain, then
    impute the wrong one by anchoring to the closest correct neighbour
    using a *local* mean gap (not the violating gap itself). Anchoring to
    the immediate neighbour avoids cascading the fix into a new violation
    further down the chain.

    Heuristics:
      V1 - if pilot is far from `1era espía atraque` (>6h), pilot is the
           typo; rebuild it as `1era espía atraque - mean(1era espía -
           pilot)`. Otherwise arribo is the typo; rebuild it as
           `pilot - mean(pilot - arribo)`.
      V2 - whichever side has the larger local anomaly (rec vs última
           espía atraque, or desp vs práctico desatraque) is the typo.
           Imputation anchors to the corresponding correct neighbour.

    All gap means are computed conditional on `(Tipo nave (agrupado),
    Terminal)` over non-violating rows, with global-mean fallback for
    empty cells.

    Input:  DataFrame with `Tipo nave (agrupado)` already populated.
    Output: DataFrame with V1 + V2 timestamps repaired in place.
    """
    df = df.copy()

    mask_v1 = df["Fecha arribo"] > df["Fecha práctico atraque"]
    mask_v2 = df["Fecha recepción nave"] > df["Fecha despacho nave"]
    n_v1, n_v2 = int(mask_v1.sum()), int(mask_v2.sum())
    if n_v1 + n_v2 == 0:
        print("  No V1/V2 ordering violations to repair")
        return df

    clean = df.loc[~(mask_v1 | mask_v2)].copy()
    g_pilot_arribo,  glob_pilot_arribo  = _conditional_local_gap(clean, "Fecha práctico atraque", "Fecha arribo")
    g_first_pilot,   glob_first_pilot   = _conditional_local_gap(clean, "1era espía atraque", "Fecha práctico atraque")
    g_rec_lastat,    glob_rec_lastat    = _conditional_local_gap(clean, "Fecha recepción nave", "Última espía atraque")
    g_pilotds_desp,  glob_pilotds_desp  = _conditional_local_gap(clean, "Fecha práctico desatraque", "Fecha despacho nave")

    # Imputed timestamps are floored to the minute to match the source-data
    # granularity (every BBDD timestamp is recorded with HH:MM:00 seconds).
    def _floor_to_minute(ts):
        return ts.floor("min")

    for idx in df.index[mask_v1]:
        r = df.loc[idx]
        type_grp, term = r["Tipo nave (agrupado)"], r["Terminal"]
        pilot, first_espia = r["Fecha práctico atraque"], r["1era espía atraque"]
        pilot_to_first_h = abs((first_espia - pilot).total_seconds()) / 3600
        if pilot_to_first_h > 6:
            gap = _lookup_gap(g_first_pilot, glob_first_pilot, type_grp, term)
            df.at[idx, "Fecha práctico atraque"] = _floor_to_minute(first_espia - pd.Timedelta(seconds=gap))
        else:
            gap = _lookup_gap(g_pilot_arribo, glob_pilot_arribo, type_grp, term)
            df.at[idx, "Fecha arribo"] = _floor_to_minute(pilot - pd.Timedelta(seconds=gap))

    for idx in df.index[mask_v2]:
        r = df.loc[idx]
        type_grp, term = r["Tipo nave (agrupado)"], r["Terminal"]
        last_at, rec   = r["Última espía atraque"], r["Fecha recepción nave"]
        desp, pilot_ds = r["Fecha despacho nave"], r["Fecha práctico desatraque"]
        rec_anomaly  = max(0.0, abs((rec - last_at).total_seconds()) / 3600 - 3)
        desp_anomaly = max(0.0, abs((pilot_ds - desp).total_seconds()) / 3600 - 3)
        if rec_anomaly >= desp_anomaly:
            gap = _lookup_gap(g_rec_lastat, glob_rec_lastat, type_grp, term)
            df.at[idx, "Fecha recepción nave"] = _floor_to_minute(last_at + pd.Timedelta(seconds=gap))
        else:
            gap = _lookup_gap(g_pilotds_desp, glob_pilotds_desp, type_grp, term)
            df.at[idx, "Fecha despacho nave"] = _floor_to_minute(pilot_ds - pd.Timedelta(seconds=gap))

    print(f"  Repaired {n_v1} V1 (arribo>pilot) and {n_v2} V2 (rec>desp) violations")
    return df


# --- Main ----------------------------------------------------------------

def main():
    """Pipeline: load, filter, rename, drop, group, save."""
    print("=" * 60)
    print("BAP - Clean Source Dataset Builder")
    print("=" * 60)

    df = load_source()

    print("\n[1/10] Filtering anomalies...")
    df = filter_anomalies(df)

    print("\n[2/10] Filtering unmoor-after-departure rows...")
    df = filter_unmoor_after_departure(df)

    print("\n[3/10] Expanding abbreviated column names...")
    df = expand_abbreviations(df)

    print("\n[4/10] Disambiguating booking date...")
    df = rename_fecha_to_solicitud(df)

    print("\n[5/10] Dropping redundant columns...")
    df = drop_redundant_columns(df)

    print("\n[6/10] Normalising terminal and site labels...")
    df = normalise_terminal_and_site(df)

    print("\n[7/10] Normalising 'Línea naviera' and 'Servicio'...")
    df = normalise_linea_and_servicio(df)

    print("\n[8/10] Adding aggregated vessel-type column...")
    df = add_vessel_type_group(df)

    print("\n[9/10] Forcing per-vessel consistent vessel-type group...")
    df = force_consistent_vessel_type_group(df)

    print("\n[10/10] Repairing V1/V2 ordering violations...")
    df = fix_ordering_violations(df)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Size:  {os.path.getsize(OUTPUT_FILE) / 1024:.0f} KB")

    print("\nFinal columns (in output order):")
    for i, col in enumerate(df.columns, start=1):
        print(f"  {i:2d}. {col}")


if __name__ == "__main__":
    main()

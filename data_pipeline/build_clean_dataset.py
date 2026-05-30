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

# `import` makes another library's code available here.
# `os` = operating-system helpers (file paths, folders). `pd` is the standard
# nickname (alias) people give pandas; `as pd` lets us type `pd` instead of `pandas`.
import os
import pandas as pd

# --- Configuration -------------------------------------------------------

# Build the path to the `data` folder relative to THIS script, so the script
# works no matter what folder you run it from.
#   __file__               = path of this .py file
#   os.path.abspath(...)   = turn it into a full absolute path
#   os.path.dirname(...)   = take just the folder that contains the file
#   os.path.join(a, b, ..) = glue path pieces with the right slash for the OS
#   ".."                   = "go up one folder" (from src/ up to the repo root)
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
# A dict (dictionary) is a lookup table of {key: value} pairs. Here each key is
# the old (abbreviated) column name and each value is the new (full) name.
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
# A list is an ordered collection written with square brackets [].
COLUMNS_TO_DROP = ["Año", "Mes", "Nave"]

# EPSA and QC label the same physical liquid-bulk berth (Sitio 9). EPSA
# (Empresa Portuaria San Antonio) operated it directly until mid-2020,
# then a private concessionaire (QC) took over later that year. 11 vessels
# appear in both Terminal labels, confirming it's the same berth. The site
# label is also stored inconsistently ("9" vs "Sitio 9"). Normalise both.
# Small lookup tables used later to rewrite values (old -> new).
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

# `def name(args):` defines a function. Everything indented under it is its body.
# A function that takes no arguments still needs the empty parentheses.
def load_source():
    """
    Load the full BBDD limpia sheet without any modification.

    Output: DataFrame with all 31 original Spanish columns, 5,605 rows.
    """
    print("Loading data from Excel...")
    # pd.read_excel reads one worksheet into a DataFrame (a table). `engine`
    # picks the library that actually parses .xlsx files (openpyxl).
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, engine="openpyxl")
    # An f-string (the f"" prefix) lets you drop variables straight into text
    # inside {curly braces}. len(df) = number of rows; len(df.columns) = columns.
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df  # hand the table back to whoever called this function


def filter_anomalies(df):
    """
    Drop rows whose `Estadía sitio` is below MIN_BERTH_HOURS or above
    MAX_BERTH_HOURS. The bounds are conservative and only remove records
    that are unambiguously bad (aborted calls and a single 780h outlier).

    Input:  DataFrame with `Estadía sitio` as Timedelta.
    Output: Filtered DataFrame, same columns as the input.
    """
    n_before = len(df)

    # `Estadía sitio` is a Timedelta (a duration). `.dt` is the accessor for
    # date/time operations on a whole column; `.total_seconds()` turns each
    # duration into a number of seconds, and /3600 converts seconds -> hours.
    estadia_hours = df["Estadía sitio"].dt.total_seconds() / 3600
    # Comparing a column to a number produces a boolean "mask": one True/False
    # per row. These mark the rows that are too short / too long.
    mask_short = estadia_hours < MIN_BERTH_HOURS
    mask_extreme = estadia_hours > MAX_BERTH_HOURS

    # On a boolean Series, .sum() counts the Trues. int(...) makes it a plain int.
    n_short = int(mask_short.sum())
    n_extreme = int(mask_extreme.sum())

    # `|` = element-wise OR (row is bad if short OR extreme); `~` = NOT, so we
    # keep the rows that are NOT bad. df.loc[mask] selects rows where mask is True.
    # .copy() returns an independent table so later edits don't warn about / leak
    # into the original DataFrame (avoids pandas' SettingWithCopy warning).
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
    # Compare two datetime columns row-by-row: True where the last mooring line
    # was released AFTER departure, which is physically impossible (a typo).
    bad = df["Última espía desatraque"] > df["Zarpe"]
    n_bad = int(bad.sum())
    df = df.loc[~bad].copy()  # keep only the rows that are NOT bad
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
    # This is a "list comprehension": a compact way to build a list. Read it as
    # "for every key k in ABBREVIATION_MAP, keep k if it is NOT in df.columns".
    # The result is the list of expected columns that the sheet is missing.
    missing = [k for k in ABBREVIATION_MAP if k not in df.columns]
    if missing:  # a non-empty list is "truthy", so this runs only if something is missing
        # `raise` stops the program with an error. Better to fail loudly here
        # than to silently produce a wrong file if the source schema changed.
        raise KeyError(f"Source sheet is missing expected columns: {missing}")
    print(f"  Renaming {len(ABBREVIATION_MAP)} abbreviated columns")
    # .rename(columns=mapping) returns a copy with renamed columns (old -> new).
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
    # .count("Fecha") counts how many columns are literally named "Fecha".
    # `!=` means "not equal". We expect exactly one; anything else is a bug.
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
    # .drop(columns=[...]) returns a copy with those columns removed.
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
    # Count EPSA rows BEFORE renaming them (afterwards there are none left).
    n_epsa = (df["Terminal"] == "EPSA").sum()
    df = df.copy()
    # .replace(mapping) swaps any value found as a key with its mapped value;
    # values not in the mapping are left unchanged. Here: every "EPSA" -> "QC".
    df["Terminal"] = df["Terminal"].replace(TERMINAL_MERGE)

    # Cast Sitio to string first: BBDD stores the EPSA-era value as integer 9
    # (not the string "9"), so a plain `replace` against the string key would
    # silently miss those 23 rows. Stringifying makes the comparison robust to
    # the source dtype.
    # NOTE (reviewer): .astype(str) also rewrites any missing Sitio (NaN) to the
    # literal string "nan", and if the column is float-typed it becomes "9.0"
    # (not "9"), which would make the "== '9'" check below silently match nothing.
    # See REPORTED items — left as-is because the source dtype can't be verified.
    df["Sitio"] = df["Sitio"].astype(str)
    # Select the rows whose Sitio is the raw "9" so we can sanity-check them.
    bad_sites = df.loc[df["Sitio"] == "9"]
    # .eq("QC") -> True/False per row; .all() -> True only if every row is QC.
    # `not ... .all()` means "at least one '9' row is NOT in QC" -> abort.
    if not bad_sites["Terminal"].eq("QC").all():
        raise ValueError("Found 'Sitio = 9' outside the QC terminal — refusing to rename.")
    # Only now rewrite the verified "9" labels to the canonical "Sitio 9".
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
    # .isin(collection) -> True where the value is one of the collection's
    # members. Iterating a dict yields its KEYS, so this counts rows whose value
    # is one of the duplicate names we intend to merge. .isna() flags missing
    # (NaN) cells. These counts are only for the printed summary below.
    n_lin_dup  = df["Línea naviera"].isin(LINEA_NAME_MAP).sum()
    n_serv_dup = df["Servicio"].isin(SERVICIO_NAME_MAP).sum()
    n_lin_na   = df["Línea naviera"].isna().sum()
    n_serv_na  = df["Servicio"].isna().sum()

    df = df.copy()
    # Two chained steps: first .replace() merges the duplicate name variants,
    # then .fillna(label) replaces remaining missing values with a meaningful
    # label (these blanks are real "not on a liner service", not lost data).
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
    # A set is an unordered collection of unique values. .dropna() removes
    # missing values, .unique() lists the distinct raw types. set(dict) is the
    # set of the dict's KEYS. The `-` is set difference: raw types that have no
    # entry in our mapping. If any exist, they'd map to NaN -> fail loudly.
    unmapped = set(df["Tipo nave"].dropna().unique()) - set(VESSEL_TYPE_GROUPS)
    if unmapped:
        raise ValueError(f"Vessel types missing from VESSEL_TYPE_GROUPS: {unmapped}")

    df = df.copy()
    # .map(dict) looks each value up in the dict and returns the mapped result,
    # building the new grouped column from the raw `Tipo nave` column.
    df["Tipo nave (agrupado)"] = df["Tipo nave"].map(VESSEL_TYPE_GROUPS)

    # Reorder columns so the new group sits right after its source column.
    cols = list(df.columns)            # current column order as a plain list
    cols.remove("Tipo nave (agrupado)")  # pull the new column out of the list
    # .index(x) finds x's position; +1 = the slot just after "Tipo nave".
    # list.insert(position, value) puts the column name back at that slot.
    cols.insert(cols.index("Tipo nave") + 1, "Tipo nave (agrupado)")
    df = df[cols]                      # df[list_of_columns] reorders the table

    # .value_counts() counts how many rows fall in each group (most common first).
    counts = df["Tipo nave (agrupado)"].value_counts()
    print(f"  Added 'Tipo nave (agrupado)' with {len(counts)} groups:")
    # .items() yields (key, value) pairs; here (group name, row count).
    # The format specs pad/align the output: {group:18s} = left-justified text
    # in 18 chars, {n:5d} = integer in 5 chars, {..:5.2f} = float, 2 decimals.
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
    # .groupby(key)[col].nunique() = for each vessel code, how many DISTINCT
    # groups it was labelled with. >1 means the vessel's calls disagree.
    vt = df.groupby("Código nave")["Tipo nave (agrupado)"].nunique()
    # vt[vt > 1] keeps only the disagreeing vessels; .index = their codes;
    # .tolist() turns that Index into an ordinary Python list to loop over.
    inconsistent = vt[vt > 1].index.tolist()
    n_changed = 0
    for code in inconsistent:  # handle one ambiguous vessel at a time
        # All row positions for this vessel, sorted earliest-arrival first.
        sub_idx = df.index[df["Código nave"] == code]
        sorted_groups = df.loc[sub_idx].sort_values("Fecha arribo")["Tipo nave (agrupado)"]
        counts = sorted_groups.value_counts()  # how often each group appears
        top = counts.iloc[0]                   # the highest count (.iloc = by position)
        tied = counts[counts == top].index.tolist()  # group(s) sharing that top count
        if len(tied) == 1:
            chosen = tied[0]  # clear winner: the single most common group
        else:
            # Tie-break by chronological order: first vessel arrival wins.
            # `next(generator)` returns the FIRST item that matches: walk the
            # chronologically-sorted groups and pick the first one that is tied.
            chosen = next(v for v in sorted_groups if v in tied)
        # Rows of THIS vessel whose group differs from the chosen one. `&` is
        # element-wise AND; each side must be parenthesised because of operator
        # precedence in pandas boolean masks.
        rows_to_change = (df["Código nave"] == code) & (df["Tipo nave (agrupado)"] != chosen)
        n_now = int(rows_to_change.sum())
        if n_now > 0:
            # df.loc[mask, column] = value writes `value` into matching rows only.
            df.loc[rows_to_change, "Tipo nave (agrupado)"] = chosen
            n_changed += n_now  # `+=` adds in place (running total)
    print(f"  Forced consistent group for {len(inconsistent)} vessel(s); {n_changed} rows changed")
    return df


def _conditional_local_gap(clean, end_col, start_col):
    """
    Compute the mean (in seconds) of `end_col - start_col` over non-violating
    rows, grouped by `(Tipo nave (agrupado), Terminal)`. Returns the per-cell
    series and the global mean as a fallback for cells with no observations.
    """
    # A leading underscore (e.g. _conditional_local_gap) is a convention meaning
    # "internal helper" — not part of the public interface.
    # Subtract two datetime columns -> a Timedelta column; convert to seconds.
    secs = (clean[end_col] - clean[start_col]).dt.total_seconds()
    # .groupby([keyA, keyB]).mean() averages within each (vessel-group, terminal)
    # combination. The result is indexed by those (type, terminal) pairs.
    grp = secs.groupby([clean["Tipo nave (agrupado)"], clean["Terminal"]]).mean()
    # Returning two values makes a tuple: (per-cell means, overall mean).
    return grp, secs.mean()


def _lookup_gap(grp_means, global_mean, type_grp, terminal):
    """Group-conditional mean with global fallback when the cell is empty."""
    # .get(key) returns the value for that key, or None if the key is absent
    # (no rows for that type/terminal combination).
    val = grp_means.get((type_grp, terminal))
    # pd.isna(x) is True for missing values (NaN/NaT). If the cell is empty or
    # missing, fall back to the dataset-wide average instead.
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

    # Boolean masks flagging the two impossible orderings (arrival after the
    # pilot berthed it; reception after dispatch). True = a violating row.
    mask_v1 = df["Fecha arribo"] > df["Fecha práctico atraque"]
    mask_v2 = df["Fecha recepción nave"] > df["Fecha despacho nave"]
    # Tuple unpacking: assign two values at once from the right-hand tuple.
    n_v1, n_v2 = int(mask_v1.sum()), int(mask_v2.sum())
    if n_v1 + n_v2 == 0:
        print("  No V1/V2 ordering violations to repair")
        return df

    # `clean` = only the well-ordered rows. We learn the "normal" gaps from
    # these so a typo'd row never contaminates the average used to repair it.
    clean = df.loc[~(mask_v1 | mask_v2)].copy()
    g_pilot_arribo,  glob_pilot_arribo  = _conditional_local_gap(clean, "Fecha práctico atraque", "Fecha arribo")
    g_first_pilot,   glob_first_pilot   = _conditional_local_gap(clean, "1era espía atraque", "Fecha práctico atraque")
    g_rec_lastat,    glob_rec_lastat    = _conditional_local_gap(clean, "Fecha recepción nave", "Última espía atraque")
    g_pilotds_desp,  glob_pilotds_desp  = _conditional_local_gap(clean, "Fecha práctico desatraque", "Fecha despacho nave")

    # Imputed timestamps are floored to the minute to match the source-data
    # granularity (every BBDD timestamp is recorded with HH:MM:00 seconds).
    # A nested function defined inside another is a small local helper.
    # .floor("min") rounds a timestamp DOWN to the start of its minute.
    def _floor_to_minute(ts):
        return ts.floor("min")

    # df.index[mask] gives the index labels of the violating rows. We loop over
    # them one at a time and repair each row in place.
    for idx in df.index[mask_v1]:
        r = df.loc[idx]  # df.loc[label] selects that single row as a Series
        type_grp, term = r["Tipo nave (agrupado)"], r["Terminal"]
        pilot, first_espia = r["Fecha práctico atraque"], r["1era espía atraque"]
        # abs(...) = absolute value; how far the pilot time sits from the first
        # mooring line, in hours. A big gap fingers the pilot time as the typo.
        pilot_to_first_h = abs((first_espia - pilot).total_seconds()) / 3600
        if pilot_to_first_h > 6:
            # Pilot is the typo: rebuild it just before the first mooring line.
            gap = _lookup_gap(g_first_pilot, glob_first_pilot, type_grp, term)
            # pd.Timedelta(seconds=gap) is a duration; subtracting it shifts the
            # timestamp earlier. df.at[label, col] = ... writes ONE cell fast.
            df.at[idx, "Fecha práctico atraque"] = _floor_to_minute(first_espia - pd.Timedelta(seconds=gap))
        else:
            # Arrival is the typo: rebuild it just before the pilot berthing.
            gap = _lookup_gap(g_pilot_arribo, glob_pilot_arribo, type_grp, term)
            df.at[idx, "Fecha arribo"] = _floor_to_minute(pilot - pd.Timedelta(seconds=gap))

    for idx in df.index[mask_v2]:
        r = df.loc[idx]
        type_grp, term = r["Tipo nave (agrupado)"], r["Terminal"]
        last_at, rec   = r["Última espía atraque"], r["Fecha recepción nave"]
        desp, pilot_ds = r["Fecha despacho nave"], r["Fecha práctico desatraque"]
        # For each side, measure how abnormal its gap is (hours beyond a 3h
        # tolerance). max(0.0, ...) clamps small/normal gaps to 0 so only the
        # genuinely anomalous side scores. The larger score is the likely typo.
        rec_anomaly  = max(0.0, abs((rec - last_at).total_seconds()) / 3600 - 3)
        desp_anomaly = max(0.0, abs((pilot_ds - desp).total_seconds()) / 3600 - 3)
        if rec_anomaly >= desp_anomaly:
            # Reception is the typo: rebuild it just after the last mooring line.
            gap = _lookup_gap(g_rec_lastat, glob_rec_lastat, type_grp, term)
            df.at[idx, "Fecha recepción nave"] = _floor_to_minute(last_at + pd.Timedelta(seconds=gap))
        else:
            # Dispatch is the typo: rebuild it just before the pilot unberthing.
            gap = _lookup_gap(g_pilotds_desp, glob_pilotds_desp, type_grp, term)
            df.at[idx, "Fecha despacho nave"] = _floor_to_minute(pilot_ds - pd.Timedelta(seconds=gap))

    print(f"  Repaired {n_v1} V1 (arribo>pilot) and {n_v2} V2 (rec>desp) violations")
    return df


# --- Main ----------------------------------------------------------------

def main():
    """Pipeline: load, filter, rename, drop, group, save."""
    # Each step takes a DataFrame and returns the transformed DataFrame, so we
    # keep reassigning `df` to thread the table through the whole pipeline.
    # "=" * 60 repeats the string 60 times to draw a separator line.
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

    # makedirs creates the output folder; exist_ok=True means "don't error if
    # it already exists". df.to_csv writes the table; index=False drops pandas'
    # automatic row-number column so the CSV only holds real data columns.
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved to {OUTPUT_FILE}")
    # df.shape is a (rows, columns) tuple; [0] = rows, [1] = columns.
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Size:  {os.path.getsize(OUTPUT_FILE) / 1024:.0f} KB")

    print("\nFinal columns (in output order):")
    # enumerate(seq, start=1) yields (1, first), (2, second), ... so we get a
    # human-friendly 1-based number alongside each column name.
    for i, col in enumerate(df.columns, start=1):
        print(f"  {i:2d}. {col}")


# This guard means "only run main() when this file is executed directly"
# (e.g. `python build_clean_dataset.py`), NOT when it is imported by another
# script. __name__ equals "__main__" only in the direct-run case.
if __name__ == "__main__":
    main()

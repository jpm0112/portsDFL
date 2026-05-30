"""
Build training dataset for BAP service time prediction.

Reads vessel call records from BBDD limpia, computes target variables and
engineered features, and outputs a clean CSV for model training.

Input:  data/BBDD limpia(1).xlsx -> sheet "Resume Naves Comerciales (4)"
Output: output/training_dataset.csv
"""

# `import` pulls in other code libraries so we can use their functions.
import sys
import os
import numpy as np          # numpy = fast numeric arrays; we use np.select below.
import pandas as pd         # pandas = tables ("DataFrames"); the main tool here.

# Add src directory to path for port_regions import.
# __file__ is the path of THIS script. os.path.abspath makes it a full path,
# os.path.dirname strips off the filename to get the folder. sys.path.insert(0, ...)
# puts that folder at the front of Python's search list so the next import works
# even when the script is run from a different working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from port_regions import get_region  # imports just the get_region function from port_regions.py

# --- Configuration ---
# os.path.join builds file paths using the right slash for the OS. ".." means
# "go up one folder", so DATA_DIR points to the sibling "data" folder next to "src".
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
INPUT_FILE = os.path.join(DATA_DIR, "BBDD limpia(1).xlsx")
SHEET_NAME = "Resume Naves Comerciales (4)"
OUTPUT_FILE = os.path.join(DATA_DIR, "training_dataset.csv")

# Column name mapping: Excel Spanish -> internal English.
# A dict ({ } ) stores key: value pairs; here it maps old column names to new ones.
COLUMN_MAP = {
    "Cód. nave": "vessel_code",
    "Nave": "vessel_name",
    "T. nave": "vessel_type",
    "TRG": "trg",
    "Puerto origen": "origin_port",
    "Puerto destino": "dest_port",
    "C. arribo proa": "draft_arrival_bow",
    "C. arribo popa": "draft_arrival_stern",
    "Agencia": "agency_raw",
    "Terminal": "terminal",
    "L. naviera": "shipping_line",
    "Servicio": "service_route",
    "Fecha práctico atraque": "pilot_boarding_datetime",
    "F. arribo": "arrival_datetime",
    "1era espía atraque": "first_mooring_datetime",
    "Última espía atraque": "last_mooring_datetime",
    "Fecha recepción nave": "reception_datetime",
    "Fecha despacho nave": "dispatch_datetime",
    "Fecha práctico desatraque": "pilot_unberthing_datetime",
    "1era espía desatraque": "first_unmooring_datetime",
    "Última espía desatraque": "last_unmooring_datetime",
    "Zarpe": "departure_datetime",
}

# Vessel type aggregation mapping
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


# `def name():` defines a function — a reusable block of code you call by name.
def load_data():
    """
    Load and rename columns from the Excel source file.

    Output: DataFrame with renamed columns
    """
    print("Loading data from Excel...")
    # pd.read_excel reads one sheet of an .xlsx file into a DataFrame (a table).
    # engine="openpyxl" is the library pandas uses to actually open .xlsx files.
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, engine="openpyxl")
    # .rename(columns=...) returns a new table with columns renamed via the dict.
    df = df.rename(columns=COLUMN_MAP)
    # An f-string (f"...") lets you drop variables inside { } directly into text.
    # len(df) = number of rows; len(df.columns) = number of columns.
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df  # hand the table back to whoever called this function.


def compute_targets(df):
    """
    Compute target variables from timestamps.

    estadia_sitio_hours: berth occupation (first mooring to last unmooring)
    tiempo_en_puerto_hours: time inside port (pilot boarding to departure)

    Input:  DataFrame with timestamp columns
    Output: DataFrame with target columns added
    """
    # Subtracting two datetime columns gives a Timedelta (a duration) per row.
    # .dt is the accessor for datetime/duration methods; .total_seconds() turns the
    # duration into seconds, and /3600 converts seconds to hours.
    # df["new_col"] = ... creates (or overwrites) a column named "new_col".
    # Berth stay: last unmooring - first mooring
    df["estadia_sitio_hours"] = (
        (df["last_unmooring_datetime"] - df["first_mooring_datetime"])
        .dt.total_seconds() / 3600
    )

    # Time inside port: departure - pilot boarding
    df["tiempo_en_puerto_hours"] = (
        (df["departure_datetime"] - df["pilot_boarding_datetime"])
        .dt.total_seconds() / 3600
    )

    # Reference: total port stay including anchorage wait (a plain local variable,
    # not a column, because the name has no df["..."] in front of it).
    estadia_puerto_hours = (
        (df["departure_datetime"] - df["arrival_datetime"])
        .dt.total_seconds() / 3600
    )
    # Pre-berth waiting time = total time in port minus the time actually at berth.
    df["espera_preatraque_hours"] = estadia_puerto_hours - df["estadia_sitio_hours"]

    print(f"  estadia_sitio_hours: mean={df['estadia_sitio_hours'].mean():.1f}, "
          f"median={df['estadia_sitio_hours'].median():.1f}")
    print(f"  tiempo_en_puerto_hours: mean={df['tiempo_en_puerto_hours'].mean():.1f}, "
          f"median={df['tiempo_en_puerto_hours'].median():.1f}")
    return df


def clean_data(df):
    """
    Remove outliers and flag quality issues.

    - Removes berth stays < 2 hours (aborted calls)
    - Removes the 780h outlier
    - Flags records where dispatch precedes reception

    Input:  DataFrame with target columns
    Output: Cleaned DataFrame with quality_flag column
    """
    n_before = len(df)

    # A comparison like (col_a < col_b) produces a True/False value per row
    # (a "boolean mask"). .astype(int) turns True->1 and False->0 so we store a flag.
    # Flag quality issues before filtering
    df["quality_flag"] = (
        (df["dispatch_datetime"] < df["reception_datetime"]).astype(int)
    )
    n_quality = df["quality_flag"].sum()  # .sum() of 1/0 flags counts the True rows.

    # Remove berth stays under 2 hours. mask_short is a True/False column.
    # NOTE: rows where estadia_sitio_hours is NaN compare as False here, so NaN
    # (un-computable) berth stays are NOT dropped by this filter.
    mask_short = df["estadia_sitio_hours"] < 2
    n_short = mask_short.sum()

    # Remove extreme outlier (780h)
    mask_extreme = df["estadia_sitio_hours"] > 500
    n_extreme = mask_extreme.sum()

    # df[mask] keeps only rows where mask is True. ~ flips True/False, and & is
    # element-wise "and", so this keeps rows that are NOT short AND NOT extreme.
    # .copy() makes an independent table to avoid pandas "SettingWithCopy" warnings.
    df = df[~mask_short & ~mask_extreme].copy()

    print(f"  Removed {n_short} records with berth stay < 2h")
    print(f"  Removed {n_extreme} records with berth stay > 500h")
    print(f"  Flagged {n_quality} records with dispatch < reception")
    print(f"  Rows: {n_before} -> {len(df)}")
    return df


def build_direct_features(df):
    """
    Extract and transform direct features from raw columns.

    - Maps vessel types to aggregated groups
    - Parses agency name from "RUT - NAME" format
    - Fills missing shipping_line and service_route with "UNKNOWN"
    - Computes draft features

    Input:  DataFrame with raw columns
    Output: DataFrame with engineered direct features
    """
    # .map(dict) looks each value up in the dict and replaces it; values not found
    # become NaN, and .fillna("Other") then replaces those NaNs with "Other".
    # Vessel type grouping
    df["vessel_type_group"] = df["vessel_type"].map(VESSEL_TYPE_GROUP).fillna("Other")

    # .apply(func) runs func on every value in the column. lambda x: ... is a tiny
    # one-line unnamed function (here, x is one agency string).
    # str(x).split(" - ", 1) splits on the FIRST " - " into at most 2 pieces;
    # [1] takes the second piece (the NAME); .strip() trims spaces. The
    # `A if cond else B` part falls back to the whole string when there's no " - ".
    # Parse agency name: "RUT - NAME" -> "NAME"
    df["agency"] = df["agency_raw"].apply(
        lambda x: str(x).split(" - ", 1)[1].strip()
        if " - " in str(x) else str(x).strip()
    )

    # .fillna(value) replaces missing (NaN) cells with the given value.
    # Fill missing categorical features
    df["shipping_line"] = df["shipping_line"].fillna("UNKNOWN")
    df["service_route"] = df["service_route"].fillna("UNKNOWN")

    # Draft = how deep the hull sits. Mean of bow+stern, max of the two, and trim
    # (stern minus bow, i.e. front-to-back tilt). df[["a","b"]].max(axis=1) takes the
    # max ACROSS the two columns per row (axis=1 = across columns, not down rows).
    # Draft features
    df["draft_arrival_mean"] = (df["draft_arrival_bow"] + df["draft_arrival_stern"]) / 2
    df["max_arrival_draft"] = df[["draft_arrival_bow", "draft_arrival_stern"]].max(axis=1)
    df["draft_trim_arrival"] = df["draft_arrival_stern"] - df["draft_arrival_bow"]

    return df


def build_temporal_features(df):
    """
    Extract time-based features from arrival datetime.

    Input:  DataFrame with arrival_datetime column
    Output: DataFrame with temporal feature columns
    """
    dt = df["arrival_datetime"]  # shorthand so we don't retype the long name.
    # The .dt accessor pulls calendar parts out of a datetime column.
    df["arrival_month"] = dt.dt.month          # 1-12
    df["arrival_day_of_week"] = dt.dt.weekday  # 0 = Monday ... 6 = Sunday
    df["arrival_hour"] = dt.dt.hour            # 0-23
    df["arrival_year"] = dt.dt.year
    df["quarter"] = dt.dt.quarter              # 1-4
    # weekday >= 5 is True for Saturday(5)/Sunday(6); astype(int) -> 1 weekend, 0 weekday.
    df["is_weekend_arrival"] = (dt.dt.weekday >= 5).astype(int)
    return df


def build_region_features(df):
    """
    Map origin and destination ports to geographic regions.

    Input:  DataFrame with origin_port and dest_port columns
    Output: DataFrame with origin_region and dest_region columns
    """
    # .apply(get_region) runs the imported get_region() on each port name to get
    # its geographic region (a named function instead of a lambda this time).
    df["origin_region"] = df["origin_port"].apply(get_region)
    df["dest_region"] = df["dest_port"].apply(get_region)

    # (col == "Other").sum() counts how many rows fell into the "Other" bucket.
    n_other_origin = (df["origin_region"] == "Other").sum()
    n_other_dest = (df["dest_region"] == "Other").sum()
    if n_other_origin > 0:
        print(f"  Warning: {n_other_origin} origin ports mapped to 'Other'")
    if n_other_dest > 0:
        print(f"  Warning: {n_other_dest} destination ports mapped to 'Other'")
    return df


def build_historical_features(df):
    """
    Build expanding-window historical features per vessel, avoiding data leakage.

    Data is sorted chronologically by first_mooring_datetime. For each visit,
    features are computed using only prior visits of the same vessel (shift by 1).

    Input:  DataFrame sorted by first_mooring_datetime
    Output: DataFrame with historical vessel features (NaN for first visits)
    """
    # Sort the WHOLE table into time order so "earlier rows = earlier visits".
    # reset_index(drop=True) renumbers rows 0,1,2,... and throws away the old index.
    # This chronological order is what makes the leakage-safe logic below work, and
    # it stays in effect for build_group_features() which runs next without re-sorting.
    df = df.sort_values("first_mooring_datetime").reset_index(drop=True)

    # --- Per-vessel historical features ---
    # .groupby("vessel_code")["estadia_sitio_hours"] splits the berth-stay column into
    # one mini-series per vessel; later operations run within each vessel separately.
    grouped = df.groupby("vessel_code")["estadia_sitio_hours"]

    # KEY ANTI-LEAKAGE PATTERN, read inside-out per vessel:
    #   .expanding().mean()  = running average over rows 0..current (grows each row)
    #   .shift(1)            = move every value DOWN one row, so each row now holds the
    #                          stat from BEFORE it (the current row's own value is
    #                          excluded; the first row of each vessel becomes NaN).
    #   .transform(...)      = apply that per-group and glue results back to original rows.
    # Result: a row only "sees" this vessel's PAST visits, never its own or future ones.
    # Expanding mean/median/std shifted by 1 (exclude current row)
    df["vessel_avg_berth_stay"] = grouped.transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df["vessel_median_berth_stay"] = grouped.transform(
        lambda x: x.expanding().median().shift(1)
    )
    df["vessel_std_berth_stay"] = grouped.transform(
        lambda x: x.expanding().std().shift(1)
    )

    # .cumcount() numbers rows within each vessel group 0,1,2,...; here it equals how
    # many PRIOR visits this vessel has had (0 = its first visit). No leakage: it counts
    # past rows only.
    # Visit count (0 = first visit, 1 = second, etc.)
    df["vessel_visit_count"] = grouped.cumcount()

    # .shift(1) within the group grabs the immediately previous visit's berth stay.
    # Last berth stay
    df["vessel_last_berth_stay"] = grouped.shift(1)

    # --- Per-vessel-terminal historical features ---
    # Same idea but grouped by vessel AND terminal (a list of keys -> finer groups).
    grouped_vt = df.groupby(["vessel_code", "terminal"])["estadia_sitio_hours"]

    df["vessel_avg_berth_stay_at_terminal"] = grouped_vt.transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df["vessel_visit_count_at_terminal"] = grouped_vt.cumcount()

    n_first = (df["vessel_visit_count"] == 0).sum()
    n_repeat = (df["vessel_visit_count"] > 0).sum()
    print(f"  First visits (NaN history): {n_first}")
    print(f"  Repeat visits (have history): {n_repeat}")
    return df


def build_group_features(df):
    """
    Build expanding-window group-level features for fallback predictions.

    Computes historical averages at (vessel_type_group, terminal), vessel_type_group,
    and terminal levels. Shifted by 1 to avoid leakage.

    Input:  DataFrame sorted chronologically
    Output: DataFrame with group-level feature columns
    """
    # NOTE: this relies on df still being in first_mooring_datetime order from
    # build_historical_features() (it is not re-sorted here). Same shift(1) trick:
    # each row's group average uses only EARLIER rows in that group, never itself.
    # Type + terminal average
    df["type_terminal_avg_stay"] = (
        df.groupby(["vessel_type_group", "terminal"])["estadia_sitio_hours"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )

    # Type average
    df["type_avg_stay"] = (
        df.groupby("vessel_type_group")["estadia_sitio_hours"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )

    # Terminal average
    df["terminal_avg_stay"] = (
        df.groupby("terminal")["estadia_sitio_hours"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )

    return df


def add_split_column(df):
    """
    Add suggested temporal train/validation/test split.

    Train: up to 2024-06-30
    Validation: 2024-07-01 to 2025-02-28
    Test: 2025-03-01 onward

    Input:  DataFrame with arrival_datetime
    Output: DataFrame with split column
    """
    # np.select(conditions, choices, default) picks, for each row, the choice tied to
    # the FIRST condition that is True (checked top to bottom), else the default.
    # Comparing a datetime column to a date string ("2024-07-01") is fine: pandas
    # parses the string to a timestamp automatically.
    # So: arrival < 2024-07-01 -> "train"; else < 2025-03-01 -> "validation"; else "test".
    conditions = [
        df["arrival_datetime"] < "2024-07-01",
        df["arrival_datetime"] < "2025-03-01",
    ]
    choices = ["train", "validation"]
    df["split"] = np.select(conditions, choices, default="test")

    # Loop over the three split names and print how many rows landed in each.
    for s in ["train", "validation", "test"]:
        print(f"  {s}: {(df['split'] == s).sum()} rows")
    return df


def select_final_columns(df):
    """
    Select and order the final columns for the output CSV.

    Groups columns into: targets, direct features, temporal, historical,
    group-level, reference columns.

    Input:  DataFrame with all computed columns
    Output: DataFrame with only the final columns, ordered
    """
    # Target variables
    targets = [
        "estadia_sitio_hours",
        "tiempo_en_puerto_hours",
    ]

    # Direct features
    direct = [
        "vessel_code",
        "vessel_name",
        "vessel_type",
        "vessel_type_group",
        "trg",
        "terminal",
        "agency",
        "shipping_line",
        "service_route",
        "origin_port",
        "dest_port",
        "origin_region",
        "dest_region",
        "draft_arrival_bow",
        "draft_arrival_stern",
        "draft_arrival_mean",
        "max_arrival_draft",
        "draft_trim_arrival",
    ]

    # Temporal features
    temporal = [
        "arrival_month",
        "arrival_day_of_week",
        "arrival_hour",
        "arrival_year",
        "quarter",
        "is_weekend_arrival",
    ]

    # Historical vessel features
    historical = [
        "vessel_avg_berth_stay",
        "vessel_median_berth_stay",
        "vessel_std_berth_stay",
        "vessel_visit_count",
        "vessel_last_berth_stay",
        "vessel_avg_berth_stay_at_terminal",
        "vessel_visit_count_at_terminal",
    ]

    # Group-level features
    group = [
        "type_terminal_avg_stay",
        "type_avg_stay",
        "terminal_avg_stay",
    ]

    # Reference columns (not features)
    reference = [
        "arrival_datetime",
        "pilot_boarding_datetime",
        "first_mooring_datetime",
        "last_unmooring_datetime",
        "departure_datetime",
        "espera_preatraque_hours",
        "quality_flag",
        "split",
    ]

    # Joining lists with + concatenates them into one ordered list of column names.
    # df[all_columns] selects only those columns, in that order (drops the rest).
    all_columns = targets + direct + temporal + historical + group + reference
    df = df[all_columns]

    # Restore original Spanish names for columns taken directly from source.
    # Derived/computed columns keep their English names.
    spanish_rename = {
        "vessel_code": "Cód. nave",
        "vessel_name": "Nave",
        "vessel_type": "T. nave",
        "trg": "TRG",
        "terminal": "Terminal",
        "shipping_line": "L. naviera",
        "service_route": "Servicio",
        "origin_port": "Puerto origen",
        "dest_port": "Puerto destino",
        "draft_arrival_bow": "C. arribo proa",
        "draft_arrival_stern": "C. arribo popa",
        "arrival_datetime": "F. arribo",
        "pilot_boarding_datetime": "Fecha práctico atraque",
        "first_mooring_datetime": "1era espía atraque",
        "last_unmooring_datetime": "Última espía desatraque",
        "departure_datetime": "Zarpe",
    }
    df = df.rename(columns=spanish_rename)
    return df


def main():
    """Main pipeline: load, compute, clean, engineer, and save."""
    print("=" * 60)
    print("BAP Service Time - Training Dataset Builder")
    print("=" * 60)

    # Step 1: Load
    df = load_data()

    # Step 2: Compute targets
    print("\nComputing target variables...")
    df = compute_targets(df)

    # Step 3: Clean
    print("\nCleaning data...")
    df = clean_data(df)

    # Step 4-5: Direct and temporal features
    print("\nBuilding direct features...")
    df = build_direct_features(df)

    print("Building temporal features...")
    df = build_temporal_features(df)

    # Step 6: Port regions
    print("Mapping port regions...")
    df = build_region_features(df)

    # Step 7: Historical vessel features
    print("\nBuilding historical vessel features...")
    df = build_historical_features(df)

    # Step 8: Group-level features
    print("Building group-level features...")
    df = build_group_features(df)

    # Step 9: Split column
    print("\nAdding temporal split...")
    df = add_split_column(df)

    # Step 10: Select and save
    print("\nSelecting final columns...")
    df = select_final_columns(df)

    # os.makedirs creates the folder; exist_ok=True means "don't error if it already exists".
    os.makedirs(DATA_DIR, exist_ok=True)
    # .to_csv writes the table to a CSV file. index=False drops the row-number column;
    # encoding="utf-8" keeps the Spanish accents (á, í, etc.) intact.
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Size: {os.path.getsize(OUTPUT_FILE) / 1024:.0f} KB")

    # Quick verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    print(f"  estadia_sitio_hours  -> mean={df['estadia_sitio_hours'].mean():.1f}, "
          f"median={df['estadia_sitio_hours'].median():.1f}, "
          f"min={df['estadia_sitio_hours'].min():.1f}, "
          f"max={df['estadia_sitio_hours'].max():.1f}")
    print(f"  tiempo_en_puerto     -> mean={df['tiempo_en_puerto_hours'].mean():.1f}, "
          f"median={df['tiempo_en_puerto_hours'].median():.1f}")
    # .isna() marks missing cells as True; .sum() counts them. These should equal the
    # number of first-ever visits (rows with no prior history to average).
    print(f"  First visits (NaN vessel_avg): "
          f"{df['vessel_avg_berth_stay'].isna().sum()}")
    print(f"  Columns: {list(df.columns)}")


# This guard means: only run main() when this file is executed directly
# (python build_training_dataset.py), NOT when another file imports it.
if __name__ == "__main__":
    main()

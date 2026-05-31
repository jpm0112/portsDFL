"""
Build training dataset for BAP service time prediction.

Reads vessel call records from BBDD limpia, computes target variables and
engineered features, and outputs a clean CSV for model training.

Input:  data/BBDD limpia(1).xlsx -> sheet "Resume Naves Comerciales (4)"
Output: output/training_dataset.csv
"""

import sys
import os
import numpy as np
import pandas as pd

# Add this script's directory to the path so port_regions imports regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from port_regions import get_region

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
INPUT_FILE = os.path.join(DATA_DIR, "BBDD limpia(1).xlsx")
SHEET_NAME = "Resume Naves Comerciales (4)"
OUTPUT_FILE = os.path.join(DATA_DIR, "training_dataset.csv")

# Column name mapping: Excel Spanish -> internal English.
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


def load_data():
    """
    Load and rename columns from the Excel source file.

    Output: DataFrame with renamed columns
    """
    print("Loading data from Excel...")
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, engine="openpyxl")
    df = df.rename(columns=COLUMN_MAP)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def compute_targets(df):
    """
    Compute target variables from timestamps.

    estadia_sitio_hours: berth occupation (first mooring to last unmooring)
    tiempo_en_puerto_hours: time inside port (pilot boarding to departure)

    Input:  DataFrame with timestamp columns
    Output: DataFrame with target columns added
    """
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

    # Reference: total port stay including anchorage wait.
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

    # Flag quality issues before filtering
    df["quality_flag"] = (
        (df["dispatch_datetime"] < df["reception_datetime"]).astype(int)
    )
    n_quality = df["quality_flag"].sum()

    # NOTE: NaN estadia_sitio_hours compares as False here, so un-computable
    # berth stays are NOT dropped by this filter.
    mask_short = df["estadia_sitio_hours"] < 2
    n_short = mask_short.sum()

    # Remove extreme outlier (780h)
    mask_extreme = df["estadia_sitio_hours"] > 500
    n_extreme = mask_extreme.sum()

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
    # Vessel type grouping (unmapped types -> "Other")
    df["vessel_type_group"] = df["vessel_type"].map(VESSEL_TYPE_GROUP).fillna("Other")

    # Parse agency name: "RUT - NAME" -> "NAME"; fall back to the whole string.
    df["agency"] = df["agency_raw"].apply(
        lambda x: str(x).split(" - ", 1)[1].strip()
        if " - " in str(x) else str(x).strip()
    )

    # Fill missing categorical features
    df["shipping_line"] = df["shipping_line"].fillna("UNKNOWN")
    df["service_route"] = df["service_route"].fillna("UNKNOWN")

    # Draft = how deep the hull sits. Mean, max, and trim (stern minus bow tilt).
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
    dt = df["arrival_datetime"]
    df["arrival_month"] = dt.dt.month          # 1-12
    df["arrival_day_of_week"] = dt.dt.weekday  # 0 = Monday ... 6 = Sunday
    df["arrival_hour"] = dt.dt.hour            # 0-23
    df["arrival_year"] = dt.dt.year
    df["quarter"] = dt.dt.quarter              # 1-4
    df["is_weekend_arrival"] = (dt.dt.weekday >= 5).astype(int)
    return df


def build_region_features(df):
    """
    Map origin and destination ports to geographic regions.

    Input:  DataFrame with origin_port and dest_port columns
    Output: DataFrame with origin_region and dest_region columns
    """
    df["origin_region"] = df["origin_port"].apply(get_region)
    df["dest_region"] = df["dest_port"].apply(get_region)

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
    # Sort chronologically so "earlier rows = earlier visits". This order makes the
    # leakage-safe logic below work, and it persists into build_group_features()
    # which runs next without re-sorting.
    df = df.sort_values("first_mooring_datetime").reset_index(drop=True)

    # --- Per-vessel historical features ---
    grouped = df.groupby("vessel_code")["estadia_sitio_hours"]

    # KEY ANTI-LEAKAGE PATTERN: .expanding() is a running stat over rows 0..current;
    # .shift(1) moves each value down one row so a row holds the stat from BEFORE it
    # (its own value excluded; first row per vessel becomes NaN). A row only ever
    # sees this vessel's PAST visits, never its own or future ones.
    df["vessel_avg_berth_stay"] = grouped.transform(
        lambda x: x.expanding().mean().shift(1)
    )
    df["vessel_median_berth_stay"] = grouped.transform(
        lambda x: x.expanding().median().shift(1)
    )
    df["vessel_std_berth_stay"] = grouped.transform(
        lambda x: x.expanding().std().shift(1)
    )

    # Visit count (0 = first visit, 1 = second, etc.; counts past rows only)
    df["vessel_visit_count"] = grouped.cumcount()

    # Last berth stay
    df["vessel_last_berth_stay"] = grouped.shift(1)

    # --- Per-vessel-terminal historical features ---
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


def _causal_group_mean(df, group_cols):
    """Leak-free running mean of estadia_sitio_hours per group.

    For each row, average over same-group visits whose outcome was ALREADY
    OBSERVED when this vessel's berthing decision is made — i.e. visits whose
    last_unmooring_datetime <= this row's first_mooring_datetime. A row's own
    visit is never included (its unmooring is after its mooring).

    This is the cross-vessel analogue of the per-vessel shift(1). FIX: a plain
    expanding().shift(1) ordered by mooring START leaks here — it would include
    other vessels that were still at berth (outcome not yet known) when this
    vessel arrived. (The per-vessel features in build_historical_features are
    NOT affected: one physical vessel cannot overlap its own prior visits, so
    those are always complete.)
    """
    cols = group_cols if isinstance(group_cols, list) else [group_cols]
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for _, idx in df.groupby(cols, sort=False).groups.items():
        sub = df.loc[idx]
        comp = sub.sort_values("last_unmooring_datetime")
        comp_t = comp["last_unmooring_datetime"].to_numpy()
        run_mean = comp["estadia_sitio_hours"].to_numpy().cumsum() / np.arange(1, len(comp) + 1)
        fm = sub["first_mooring_datetime"].to_numpy()
        # k = number of same-group visits completed by this row's mooring start.
        k = np.searchsorted(comp_t, fm, side="right")
        out.loc[sub.index] = np.where(k > 0, run_mean[np.clip(k - 1, 0, len(run_mean) - 1)], np.nan)
    return out


def build_group_features(df):
    """
    Build leak-free group-level features for fallback predictions.

    Historical averages at (vessel_type_group, terminal), vessel_type_group, and
    terminal levels, each using only visits that had COMPLETED before the current
    vessel's berthing decision (see _causal_group_mean).

    Input:  DataFrame with first_mooring_datetime / last_unmooring_datetime
    Output: DataFrame with group-level feature columns
    """
    df["type_terminal_avg_stay"] = _causal_group_mean(df, ["vessel_type_group", "terminal"])
    df["type_avg_stay"] = _causal_group_mean(df, "vessel_type_group")
    df["terminal_avg_stay"] = _causal_group_mean(df, "terminal")
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
    # arrival < 2024-07-01 -> "train"; else < 2025-03-01 -> "validation"; else "test".
    conditions = [
        df["arrival_datetime"] < "2024-07-01",
        df["arrival_datetime"] < "2025-03-01",
    ]
    choices = ["train", "validation"]
    df["split"] = np.select(conditions, choices, default="test")

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

    os.makedirs(DATA_DIR, exist_ok=True)
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
    # Should equal the number of first-ever visits (no prior history to average).
    print(f"  First visits (NaN vessel_avg): "
          f"{df['vessel_avg_berth_stay'].isna().sum()}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()

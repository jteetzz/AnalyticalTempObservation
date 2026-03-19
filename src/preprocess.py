from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT      = Path(__file__).resolve().parent.parent
DATA_RAW_DIR      = PROJECT_ROOT / "data_raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data_processed"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Columns that are feature inputs to the model (not target, not identifiers)
FEATURE_COLS = [
    "co2", "ch4", "n2o",
    "co2_growth", "ch4_growth", "n2o_growth",
    "co2_ma12", "ch4_ma12", "n2o_ma12",
    "tsi", "tsi_anomaly",
    "aerosol_optical_depth", "aerosol_log", "aerosol_ma12",
    "owid_co2", "owid_co2_luc", "owid_primary_energy",
    "time_since_baseline",
    "month_sin", "month_cos",
]

TARGET_COL = "temp_anomaly"
ID_COLS    = ["year", "month", "region"]


def preprocess() -> None:
    input_path  = DATA_RAW_DIR      / "merged_monthly_regional.csv"   # matches collect_data.py
    output_path = DATA_PROCESSED_DIR / "cleaned_monthly_regional.csv"

    # 1. Load 
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows from {input_path.name}")

    # 2. Sort correctly for monthly data 
    df = df.sort_values(["region", "year", "month"]).reset_index(drop=True)

    # 3. Validate no duplicate (year, month, region) 
    dupes = df.duplicated(subset=["year", "month", "region"]).sum()
    if dupes > 0:
        print(f"  Warning: dropping {dupes} duplicate (year, month, region) rows")
        df = df.drop_duplicates(subset=["year", "month", "region"]).reset_index(drop=True)

    # 4. Drop rows missing the target 
    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"  Dropped {before - len(df)} rows missing target '{TARGET_COL}'")

    # 5. Fill remaining nulls 
    # TSI: interpolate within region, then fill any edge gaps with median
    if "tsi" in df.columns:
        df["tsi"] = (
            df.groupby("region")["tsi"]
            .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
        )
        tsi_median = df["tsi"].median()
        df["tsi"] = df["tsi"].fillna(tsi_median)
        print(f"  TSI nulls remaining after fill: {df['tsi'].isna().sum()}")

    # Aerosol: already filled to 0 in collect_data.py but guard here too
    for col in ["aerosol_optical_depth", "aerosol_log", "aerosol_ma12"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # OWID yearly columns sparse at edges and forward/back fill within region
    for col in ["owid_co2", "owid_co2_luc", "owid_primary_energy"]:
        if col in df.columns:
            df[col] = (
                df.groupby("region")[col]
                .transform(lambda s: s.ffill().bfill())
            )

    # Growth/MA cols: first 12 months per region are NaN by construction
    # Drop those warmup rows (already done in collect_data.py, but be safe)
    growth_cols = [c for c in ["co2_growth","ch4_growth","n2o_growth"] if c in df.columns]
    if growth_cols:
        before = len(df)
        df = df.dropna(subset=growth_cols).reset_index(drop=True)
        print(f"  Dropped {before - len(df)} warmup rows (growth cols null)")

    # 6. Normalize features = z-score 
    # Keep original columns; add _norm suffix for scaled versions
    existing_features = [c for c in FEATURE_COLS if c in df.columns]
    norm_stats = {}

    for col in existing_features:
        mean = df[col].mean()
        std  = df[col].std()
        if std == 0 or np.isnan(std):
            df[f"{col}_norm"] = 0.0
        else:
            df[f"{col}_norm"] = (df[col] - mean) / std
        norm_stats[col] = {"mean": mean, "std": std}

    # Save normalization stats so they can be inverted at prediction time
    stats_df = pd.DataFrame(norm_stats).T.reset_index().rename(columns={"index": "feature"})
    stats_path = DATA_PROCESSED_DIR / "normalization_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  Normalization stats saved to {stats_path.name}")

    # 7. Final summary and save 
    df.to_csv(output_path, index=False)

    print("\nPreprocessing complete.")
    print(f"  Saved to  : {output_path}")
    print(f"  Rows      : {len(df):,}")
    print(f"  Columns   : {len(df.columns)}")
    print(f"  Year range: {df['year'].min()}–{df['year'].max()}")
    print(f"  Regions   : {sorted(df['region'].unique())}")
    print(f"\n  Null counts in feature columns:")
    null_summary = df[existing_features].isna().sum()
    print(null_summary[null_summary > 0].to_string() if null_summary.any() else "    None ✓")
    print("\nFirst 3 rows:")
    print(df[ID_COLS + [TARGET_COL] + existing_features[:4]].head(3).to_string(index=False))


if __name__ == "__main__":
    preprocess()

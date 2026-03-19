from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT       = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data_processed"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def feature_engineering() -> None:
    input_path  = DATA_PROCESSED_DIR / "cleaned_monthly_regional.csv"
    output_path = DATA_PROCESSED_DIR / "features_monthly_regional.csv"

    df = pd.read_csv(input_path)

    # Sort correctly for monthly data 
    df = df.sort_values(["region", "year", "month"]).reset_index(drop=True)

    # 1. Lag features (1-month) 
    # Useful for autoregressive signal; shift(1) within each region
    for col in ["co2", "ch4", "n2o", "tsi"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("region")[col].shift(1)

    # 2. Percentage growth rates (month-over-month) 
    # pct_change is more comparable across gases than absolute diff
    for gas in ["co2", "ch4", "n2o"]:
        if gas in df.columns:
            df[f"{gas}_pct_growth"] = (
                df.groupby("region")[gas]
                .transform(lambda s: s.pct_change(1) * 100)  # in %
            )

    # 3. Interaction term: CO2 * CH4 
    # Both are in ppm — same-unit multiplication is physically meaningful
    # and captures combined radiative forcing signal
    if "co2" in df.columns and "ch4" in df.columns:
        df["co2_ch4_interaction"] = df["co2"] * df["ch4"]

    # 4. Human vs natural forcing index 
    # Compare anthropogenic forcing (CO2 normalized) vs solar forcing (TSI anomaly)
    # Use z-score normalized versions so units are comparable
    if "co2_norm" in df.columns and "tsi_anomaly_norm" in df.columns:
        df["human_vs_natural"] = df["co2_norm"] - df["tsi_anomaly_norm"]
    elif "co2" in df.columns and "tsi_anomaly" in df.columns:
        # Fall back to manual normalization if _norm cols not present
        co2_z = (df["co2"] - df["co2"].mean()) / df["co2"].std()
        tsi_z = (df["tsi_anomaly"] - df["tsi_anomaly"].mean()) / df["tsi_anomaly"].std()
        df["human_vs_natural"] = co2_z - tsi_z

    # 5. Volcanic activity flag
    # Aerosol optical depth > 0.05 indicates significant volcanic stratospheric
    # loading (e.g. Pinatubo 1991 peaked near 0.15)
    if "aerosol_optical_depth" in df.columns:
        VOLCANIC_THRESHOLD = 0.05
        df["volcanic_flag"] = (df["aerosol_optical_depth"] > VOLCANIC_THRESHOLD).astype(int)
        print(f"  Volcanic flag: {df['volcanic_flag'].sum()} months flagged "
              f"(threshold = {VOLCANIC_THRESHOLD})")

    # 6. Drop warmup rows where lag/pct features are null 
    lag_cols = [c for c in df.columns if c.endswith("_lag1") or c.endswith("_pct_growth")]
    existing_lag_cols = [c for c in lag_cols if c in df.columns]
    before = len(df)
    df = df.dropna(subset=existing_lag_cols).reset_index(drop=True)
    print(f"  Dropped {before - len(df)} warmup rows after lag/pct features")

    # 7. Final check 
    print(f"\n  Remaining nulls per column:")
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts) == 0:
        print("    None ✓")
    else:
        print(null_counts.to_string())

    # 8. Save 
    df.to_csv(output_path, index=False)

    print("\nFeature engineering complete.")
    print(f"  Saved to  : {output_path}")
    print(f"  Rows      : {len(df):,}")
    print(f"  Columns   : {len(df.columns)}")
    print(f"  New cols added this step: co2_lag1, ch4_lag1, n2o_lag1, tsi_lag1, "
          f"co2_pct_growth, ch4_pct_growth, n2o_pct_growth, "
          f"co2_ch4_interaction, human_vs_natural, volcanic_flag")
    print("\nFirst 3 rows (key columns):")
    preview_cols = ["year", "month", "region", "temp_anomaly",
                    "co2", "co2_lag1", "co2_pct_growth",
                    "co2_ch4_interaction", "human_vs_natural", "volcanic_flag"]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head(3).to_string(index=False))


if __name__ == "__main__":
    feature_engineering()

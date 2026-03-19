from __future__ import annotations

import io
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,application/xml,text/xml,*/*",
}

# Source URLs
NOAA_CO2_MONTHLY_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.csv"
NOAA_CH4_MONTHLY_URL = "https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_mm_gl.csv"
NOAA_N2O_MONTHLY_URL = "https://gml.noaa.gov/webdata/ccgg/trends/n2o/n2o_mm_gl.csv"

# NASA provides two URL formats for the same file, so we try both
NASA_GISTEMP_MONTHLY_URLS = [
    "https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts%2BdSST.csv",
    "https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv",
]
NASA_GISTEMP_MONTHLY_GLOBAL_URLS = [
    "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts%2BdSST.csv",
    "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
]

TSI_YEARLY_CATALOG_XML     = "https://www.ncei.noaa.gov/thredds/catalog/cdr-total-solar-irradiance/yearly/catalog.xml"
TSI_YEARLY_FILESERVER_BASE = "https://www.ncei.noaa.gov/thredds/fileServer/cdr-total-solar-irradiance/yearly/"

AEROSOL_ASCII_URL = "https://data.giss.nasa.gov/modelforce/strataer/tau.line_2012.12.txt"
OWID_CO2_URL      = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"


def fetch_text(url: str, timeout: int = 180) -> str:
    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text


def fetch_binary(url: str, timeout: int = 180) -> bytes:
    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.content


def try_urls(urls: list[str]) -> str:
    last_err = None
    for url in urls:
        try:
            text = fetch_text(url)
            print(f"  loaded from {url}")
            return text
        except requests.RequestException as e:
            last_err = e
            print(f"  failed {url}: {e}")
    raise RuntimeError(f"All URLs failed. Last error: {last_err}")


def _load_noaa_monthly_gas(url: str, gas_name: str) -> pd.DataFrame:
# All three NOAA gas files share the same format 
    raw = fetch_text(url)
    data_lines = [l for l in raw.splitlines() if not l.strip().startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    df.columns = [c.strip().lower() for c in df.columns]

    year_col  = next(c for c in df.columns if "year"    in c)
    month_col = next(c for c in df.columns if "month"   in c)
    avg_col   = next(c for c in df.columns if "average" in c and "unc" not in c)

    df = df[[year_col, month_col, avg_col]].copy()
    df.columns = ["year", "month", gas_name]
    df["year"]   = pd.to_numeric(df["year"],   errors="coerce")
    df["month"]  = pd.to_numeric(df["month"],  errors="coerce")
    df[gas_name] = pd.to_numeric(df[gas_name], errors="coerce")

    # NOAA flags missing values as -999.99
    df.loc[df[gas_name] < -900, gas_name] = np.nan

    df = df.dropna().copy()
    df["year"]  = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    return df


def load_noaa_monthly_ghg() -> pd.DataFrame:
    print("Loading NOAA monthly CO2...")
    co2 = _load_noaa_monthly_gas(NOAA_CO2_MONTHLY_URL, "co2")
    print("Loading NOAA monthly CH4...")
    ch4 = _load_noaa_monthly_gas(NOAA_CH4_MONTHLY_URL, "ch4")
    print("Loading NOAA monthly N2O...")
    n2o = _load_noaa_monthly_gas(NOAA_N2O_MONTHLY_URL, "n2o")

    # CO2 goes back to 1979, CH4 to ~1983, N2O to ~2001.
    # We left-join on CO2 so the early years aren't dropped, then interpolate
    # the gaps in CH4 and N2O — both gases change slowly enough that linear
    # interpolation is a reasonable fill for the missing early months.
    df = co2.merge(ch4, on=["year", "month"], how="left")
    df = df.merge(n2o, on=["year", "month"], how="left")
    df = df.sort_values(["year", "month"]).reset_index(drop=True)

    for gas in ["ch4", "n2o"]:
        df[gas] = df[gas].interpolate(method="linear", limit_direction="both")

    print(f"  GHG rows: {len(df):,}, {df['year'].min()}–{df['year'].max()}")
    print(f"  CH4 nulls remaining: {df['ch4'].isna().sum()}")
    print(f"  N2O nulls remaining: {df['n2o'].isna().sum()}")
    return df


# Month column names as they appear in the GISTEMP file
MONTH_COLS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Maps our region labels to the column names in the GISTEMP zonal file
REGION_MAP = {
    "Global":                "Glob",
    "Northern_Hemisphere":   "NHem",
    "Southern_Hemisphere":   "SHem",
    "Tropics":               "24N-24S",
    "Northern_Extratropics": "24N-90N",
    "Southern_Extratropics": "90S-24S",
}


def load_gistemp_monthly_global() -> pd.DataFrame:
# Loads the GISTEMP global monthly table (GLB.Ts+dSST.csv)
    print("Loading NASA GISTEMP monthly global...")
    raw = try_urls(NASA_GISTEMP_MONTHLY_GLOBAL_URLS)
    lines = raw.splitlines()

    header_idx = next(i for i, l in enumerate(lines) if l.startswith("Year,"))
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    df.columns = [c.strip() for c in df.columns]

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    month_cols_found = [c for c in MONTH_COLS if c in df.columns]
    df = df[["Year"] + month_cols_found].copy()

    for c in month_cols_found:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] == 9999, c] = np.nan  # GISTEMP missing marker

    month_name_to_num = {m: i + 1 for i, m in enumerate(MONTH_COLS)}
    long = df.melt(id_vars="Year", value_vars=month_cols_found,
                   var_name="month_name", value_name="temp_global")
    long["month"] = long["month_name"].str.strip().str[:3].map(month_name_to_num)
    long = long.rename(columns={"Year": "year"})
    long = long[["year", "month", "temp_global"]].dropna().copy()
    long["year"]  = long["year"].astype(int)
    long["month"] = long["month"].astype(int)

    print(f"  GISTEMP global monthly: {len(long):,} rows, {long['year'].min()}–{long['year'].max()}")
    return long.sort_values(["year", "month"]).reset_index(drop=True)


def load_gistemp_zonal_annual() -> pd.DataFrame:
# Loads the GISTEMP zonal annual table (ZonAnn.Ts+dSST.csv)
    print("Loading NASA GISTEMP annual zonal...")
    raw = try_urls(NASA_GISTEMP_MONTHLY_URLS)
    lines = raw.splitlines()

    header_idx = next(i for i, l in enumerate(lines) if l.startswith("Year,"))
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    df.columns = [c.strip() for c in df.columns]

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)

    zone_cols = [c for c in REGION_MAP.values() if c in df.columns]
    df = df[["Year"] + zone_cols].copy()
    for c in zone_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] == 9999, c] = np.nan

    df = df.rename(columns={"Year": "year"})
    print(f"  GISTEMP annual zonal: {len(df):,} rows")
    return df


def build_monthly_regional_temp(
    monthly_global: pd.DataFrame,
    annual_zonal: pd.DataFrame,
) -> pd.DataFrame:
    ann_global = (
        monthly_global.groupby("year")["temp_global"].mean()
        .rename("ann_global").reset_index()
    )

    records = []
    for region_label, zone_col in REGION_MAP.items():
        if zone_col not in annual_zonal.columns:
            print(f"  Warning: zone column '{zone_col}' not found, skipping '{region_label}'")
            continue

        zone_ann = annual_zonal[["year", zone_col]].dropna().copy()
        zone_ann = zone_ann.merge(ann_global, on="year", how="inner")
        zone_ann["offset"] = zone_ann[zone_col] - zone_ann["ann_global"]

        tmp = monthly_global.merge(zone_ann[["year", "offset"]], on="year", how="inner")
        tmp["temp_anomaly"] = tmp["temp_global"] + tmp["offset"]
        tmp["region"] = region_label
        records.append(tmp[["year", "month", "region", "temp_anomaly"]])

    df = pd.concat(records, ignore_index=True)
    df = df.dropna(subset=["temp_anomaly"]).reset_index(drop=True)
    print(f"  Regional monthly temp: {len(df):,} rows, regions: {sorted(df['region'].unique())}")
    return df


def get_latest_tsi_yearly_file_url() -> str:
# Reads the THREDDS catalog XML to find the most recent yearly TSI file
    xml_text = fetch_text(TSI_YEARLY_CATALOG_XML)
    root = ET.fromstring(xml_text)
    ns = {"cat": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"}
    names = [
        ds.attrib["name"]
        for ds in root.findall(".//cat:dataset", ns)
        if ds.attrib.get("name", "").endswith(".nc") and "yearly" in ds.attrib.get("name", "")
    ]
    if not names:
        raise ValueError("No yearly TSI NetCDF file found in THREDDS catalog.")
    return TSI_YEARLY_FILESERVER_BASE + sorted(names)[-1]


def load_tsi_monthly() -> pd.DataFrame:
# Downloads the yearly TSI NetCDF from NOAA and interpolates it to monthly.

    print("Loading NOAA TSI (yearly, will interpolate to monthly)...")
    url = get_latest_tsi_yearly_file_url()
    print(f"  TSI file: {url}")
    content = fetch_binary(url)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        ds = xr.open_dataset(tmp_path)
        data_var = "TSI" if "TSI" in ds.data_vars else list(ds.data_vars)[0]
        df = ds[[data_var]].to_dataframe().reset_index()
        ds.close()
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    time_col = "time" if "time" in df.columns else [c for c in df.columns if c != data_var][0]
    df = df[[time_col, data_var]].copy()
    df.columns = ["date_raw", "tsi"]
    df["year"] = df["date_raw"].apply(lambda x: getattr(x, "year", np.nan))
    df["tsi"]  = pd.to_numeric(df["tsi"], errors="coerce")
    df.loc[df["tsi"] < 0, "tsi"] = np.nan
    df = df.dropna(subset=["year", "tsi"]).copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year").drop_duplicates("year").reset_index(drop=True)

    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    months = pd.DataFrame(
        [(y, m) for y in range(year_min, year_max + 1) for m in range(1, 13)],
        columns=["year", "month"]
    )
    df["frac_year"]     = df["year"] + 0.5
    months["frac_year"] = months["year"] + (months["month"] - 0.5) / 12

    months["tsi"] = np.interp(
        months["frac_year"].values,
        df["frac_year"].values,
        df["tsi"].values,
    )
    months = months[["year", "month", "tsi"]]
    print(f"  TSI monthly: {len(months):,} rows, {months['year'].min()}–{months['year'].max()}")
    return months


def load_aerosol_monthly() -> pd.DataFrame:
# The NASA GISS aerosol file already stores monthly values 
    print("Loading NASA aerosol optical depth...")
    raw = fetch_text(AEROSOL_ASCII_URL)

    rows = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            year = int(float(parts[0]))
        except ValueError:
            continue
        for m_idx, token in enumerate(parts[1:13], start=1):
            try:
                val = float(token)
                if val < 0:
                    val = np.nan
            except ValueError:
                val = np.nan
            rows.append({"year": year, "month": m_idx, "aerosol_optical_depth": val})

    df = pd.DataFrame(rows)
    df["aerosol_optical_depth"] = pd.to_numeric(df["aerosol_optical_depth"], errors="coerce")
    df = df.dropna(subset=["aerosol_optical_depth"]).copy()
    df["year"]  = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    print(f"  Aerosol monthly: {len(df):,} rows, {df['year'].min()}–{df['year'].max()}")
    return df


def load_owid_monthly() -> pd.DataFrame:
 # OWID only publishes yearly totals, so we expand each year into 12 identical monthly rows
    print("Loading OWID annual data and expanding to monthly...")
    df = pd.read_csv(OWID_CO2_URL)

    keep = ["country", "year", "co2", "co2_including_luc", "primary_energy_consumption"]
    df = df[df["country"] == "World"][keep].copy()
    df = df.rename(columns={
        "co2":                        "owid_co2",
        "co2_including_luc":          "owid_co2_luc",
        "primary_energy_consumption": "owid_primary_energy",
    })
    for c in ["owid_co2", "owid_co2_luc", "owid_primary_energy"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
    df = df.sort_values("year").drop_duplicates("year").reset_index(drop=True)

    expanded = []
    for _, row in df.iterrows():
        for m in range(1, 13):
            expanded.append({
                "year":                int(row["year"]),
                "month":               m,
                "owid_co2":            row["owid_co2"],
                "owid_co2_luc":        row["owid_co2_luc"],
                "owid_primary_energy": row["owid_primary_energy"],
            })

    out = pd.DataFrame(expanded)
    print(f"  OWID monthly: {len(out):,} rows, {out['year'].min()}–{out['year'].max()}")
    return out


def build_merged_dataset(start_year: int = 1979, end_year: int = 2024) -> pd.DataFrame:
    ghg   = load_noaa_monthly_ghg()
    glb_m = load_gistemp_monthly_global()
    zon_a = load_gistemp_zonal_annual()
    temp  = build_monthly_regional_temp(glb_m, zon_a)
    tsi   = load_tsi_monthly()

    try:
        aerosol = load_aerosol_monthly()
        aerosol_ok = True
    except Exception as e:
        print(f"  Warning: aerosol load failed ({e}), will fill column with NaN")
        aerosol_ok = False

    owid = load_owid_monthly()

    # Use temperature as the backbone since it has the longest record.
    # Everything else is left-joined so missing data becomes NaN rather than dropped rows.
    merged = temp.merge(ghg,  on=["year", "month"], how="inner")
    merged = merged.merge(tsi,  on=["year", "month"], how="left")
    merged = merged.merge(owid, on=["year", "month"], how="left")

    if aerosol_ok:
        merged = merged.merge(aerosol, on=["year", "month"], how="left")
    else:
        merged["aerosol_optical_depth"] = np.nan

    merged = merged[(merged["year"] >= start_year) & (merged["year"] <= end_year)].copy()
    merged = merged.sort_values(["year", "month", "region"]).reset_index(drop=True)
    merged = merged.drop_duplicates(subset=["year", "month", "region"]).reset_index(drop=True)

    print(f"\nMerged dataset: {len(merged):,} rows")
    print(f"  Regions    : {sorted(merged['region'].unique())}")
    print(f"  Year range : {merged['year'].min()}–{merged['year'].max()}")
    print(f"  Columns    : {list(merged.columns)}")
    return merged


def engineer_features(df: pd.DataFrame, baseline_year: int = 1979) -> pd.DataFrame:
    df = df.copy()

    df["decimal_year"]        = df["year"] + (df["month"] - 1) / 12
    df["time_since_baseline"] = df["decimal_year"] - baseline_year

    # Sort by region first so rolling windows don't bleed across regions
    df = df.sort_values(["region", "year", "month"]).reset_index(drop=True)

    for gas in ["co2", "ch4", "n2o"]:
        df[f"{gas}_growth"] = df.groupby("region")[gas].diff(12)
        df[f"{gas}_ma12"]   = (
            df.groupby("region")[gas]
            .transform(lambda x: x.rolling(12, min_periods=6).mean())
        )

    df["temp_ma12"] = (
        df.groupby("region")["temp_anomaly"]
        .transform(lambda x: x.rolling(12, min_periods=6).mean())
    )

    if "aerosol_optical_depth" in df.columns:
        df["aerosol_log"]  = np.log1p(df["aerosol_optical_depth"].clip(lower=0))
        df["aerosol_ma12"] = (
            df.groupby("region")["aerosol_optical_depth"]
            .transform(lambda x: x.rolling(12, min_periods=6).mean())
        )

    if "tsi" in df.columns:
        df["tsi_anomaly"] = df["tsi"] - df["tsi"].mean()

    # Encode month as a cyclical feature so January and December are close together
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Fill post-2012 aerosol nulls with 0 — no major volcanic eruptions after 2012
    for col in ["aerosol_optical_depth", "aerosol_log", "aerosol_ma12"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Drop the first 12 months per region where growth rate can't be computed
    df = df.dropna(subset=["co2_growth", "ch4_growth", "n2o_growth"])

    print(f"Feature engineering done. Shape: {df.shape}")
    return df


def save_outputs(df: pd.DataFrame) -> None:
    path = DATA_RAW_DIR / "merged_monthly_regional.csv"
    df.to_csv(path, index=False)
    print(f"\nSaved to: {path}")
    print(f"  Rows    : {len(df):,}")
    print(f"  Regions : {df['region'].nunique()}")
    print(f"  Years   : {df['year'].min()}–{df['year'].max()}")
    print(f"  Columns : {list(df.columns)}")


def main() -> None:
    df = build_merged_dataset(start_year=1979, end_year=2024)
    df = engineer_features(df)
    save_outputs(df)


if __name__ == "__main__":
    main()

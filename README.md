# Identifying Root Causes of Global Temperature Change
### Multi-Source Data Integration and Regression Feature Ranking

This project builds a regression-based analytical pipeline to identify the most influential drivers of global temperature change. Environmental data is collected from multiple independent scientific sources, merged into a unified monthly dataset, and used to train regression models. Feature importance techniques are then applied to determine which factors — greenhouse gases, solar irradiance, volcanic aerosols, or energy consumption — contribute most to observed temperature variation.

---

## Project Structure

```
AnalyticalTempObservation/
│
├── src/
│   ├── load_data.py          # Downloads and merges all source data
│   ├── preprocess.py            # Cleans, validates, and normalizes the dataset
│   ├── feature_engineering.py   # Adds lag, interaction, and derived features
│   ├── evaluate.py              # Trains models and ranks feature importance
│   └── visualize.py             # Generates all plots and charts
│
├── data_raw/
│   └── merged_monthly_regional.csv       # Output of collect_data.py
│
├── data_processed/
│   ├── cleaned_monthly_regional.csv      # Output of preprocess.py
│   ├── features_monthly_regional.csv     # Output of feature_engineering.py
│   └── normalization_stats.csv           # Z-score parameters per feature
│
├── results/
│   ├── model_metrics.csv
│   ├── linear_regression_coefficients.csv
│   ├── RandomForest_feature_importance.csv
│   ├── RandomForest_permutation_importance.csv
│   ├── GradientBoosting_feature_importance.csv
│   ├── GradientBoosting_permutation_importance.csv
│   └── *.png                             # All generated plots
│
├── requirements.txt
└── README.md
```

---

## Data Sources

| Source | Data | URL |
|--------|------|-----|
| NOAA GML | Monthly CO₂, CH₄, N₂O mole fractions | https://gml.noaa.gov/ccgg/trends |
| NASA GISTEMP v4 | Monthly global + zonal temperature anomalies | https://data.giss.nasa.gov/gistemp |
| NOAA CDR | Yearly total solar irradiance (NetCDF) | https://www.ncei.noaa.gov/data/total-solar-irradiance |
| NASA GISS | Stratospheric aerosol optical depth (volcanic proxy) | https://data.giss.nasa.gov/modelforce/strataer |
| Our World in Data | Annual CO₂ emissions and primary energy consumption | https://ourworldindata.org/co2-emissions |

### Regions
- Global
- Northern Hemisphere
- Southern Hemisphere
- Tropics (24°S–24°N)
- Northern Extratropics (24°N–90°N)
- Southern Extratropics (90°S–24°S)

---

## Installation

**Python 3.12 or higher is recommended.**

1. Clone the repository:
```bash
git clone https://github.com/your-username/AnalyticalTempObservation.git
cd AnalyticalTempObservation
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### requirements.txt
```
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.4
seaborn==0.13.2
scikit-learn==1.5.1
requests==2.32.3
xarray==2025.1.2
netCDF4==1.7.2
scipy
```

> **Note:** `scipy` is required by `visualize.py` for OLS regression lines but is not pinned — install with `pip install scipy` if not pulled in automatically.

---

## How to Run

Scripts must be run **in order** from the `src/` directory. Each script reads the output of the previous one.

### Step 1 — Collect and merge data
```bash
python src/load_data.py
```
Downloads data from all five sources over the internet and saves the merged monthly dataset to `data_raw/merged_monthly_regional.csv`.

---

### Step 2 — Preprocess
```bash
python src/preprocess.py
```
- Validates and deduplicates rows
- Interpolates TSI gaps within each region
- Fills post-2012 aerosol nulls with `0.0` (no significant volcanic forcing after 2012)
- Forward-fills sparse OWID yearly values
- Z-score normalizes all feature columns (saves stats to `normalization_stats.csv`)
- Drops warmup rows where growth-rate columns are null

Output: `data_processed/cleaned_monthly_regional.csv`

---

### Step 3 — Feature engineering
```bash
python src/feature_engineering.py
```
Adds the following features on top of what `load_data.py` already produced:

| Feature | Description |
|---------|-------------|
| `co2_lag1`, `ch4_lag1`, `n2o_lag1`, `tsi_lag1` | 1-month lag (autoregressive signal) |
| `co2_pct_growth`, `ch4_pct_growth`, `n2o_pct_growth` | Month-over-month % change |
| `co2_ch4_interaction` | CO₂ × CH₄ interaction term (both in ppm/ppb) |
| `human_vs_natural` | Normalized CO₂ minus normalized TSI anomaly |
| `volcanic_flag` | Binary flag: aerosol optical depth > 0.05 |

Output: `data_processed/features_monthly_regional.csv`

---

### Step 4 — Train models and rank features
```bash
python src/evaluate.py
```
Trains three regression models using a **temporal train/test split** (earliest 70% of months → train, latest 30% → test) to avoid data leakage:

| Model | Feature set | Importance method |
|-------|-------------|-------------------|
| Linear Regression | Z-scored `_norm` columns | Standardized coefficients |
| Random Forest (300 trees) | Raw feature columns | Built-in + permutation importance |
| Gradient Boosting (300 trees, lr=0.05) | Raw feature columns | Built-in + permutation importance |

Outputs saved to `results/`:
- `model_metrics.csv` — RMSE and R² for all three models
- `*_coefficients.csv` / `*_importance.csv` — ranked feature importance tables
- `*.png` — one bar chart per model

---

### Step 5 — Visualize
```bash
python src/visualize.py
```
Generates all required plots to `results/`:

| File | Description |
|------|-------------|
| `ghg_vs_temperature.png` | CO₂, CH₄, N₂O each on dual-axis with temperature |
| `natural_forcings_vs_temperature.png` | TSI and aerosol optical depth vs temperature |
| `regional_temperature_trends.png` | Annual mean anomaly per region over time |
| `scatter_top_features.png` | Scatter + OLS fit for CO₂, CH₄, TSI, aerosol |
| `feature_importance_comparison.png` | Side-by-side bar charts from all three models |

> **Note:** Run `evaluate.py` before `visualize.py` so the importance CSVs exist. The script skips importance plots gracefully if they are missing.

---

## Engineered Features (load_data.py)

These are computed during data collection and carried through the full pipeline:

| Feature | Description |
|---------|-------------|
| `co2_growth`, `ch4_growth`, `n2o_growth` | 12-month absolute difference per region |
| `co2_ma12`, `ch4_ma12`, `n2o_ma12` | 12-month rolling mean per region |
| `temp_ma12` | 12-month rolling mean of temperature anomaly |
| `aerosol_log` | log1p transform of aerosol optical depth |
| `aerosol_ma12` | 12-month rolling mean of aerosol optical depth |
| `tsi_anomaly` | TSI minus its long-run mean |
| `time_since_baseline` | Decimal years since 1979 |
| `month_sin`, `month_cos` | Cyclical encoding of calendar month |

---

## Notes on Missing Data

| Column | Issue | Fix applied |
|--------|-------|-------------|
| `aerosol_optical_depth` | Source only covers up to 2012 | Filled with `0.0` — no major eruptions post-2012 |
| `tsi` | Yearly source interpolated to monthly | Linear interpolation; edge gaps filled with median |
| `owid_*` | Yearly source expanded to monthly | Forward-fill then back-fill within each region |
| Growth/MA cols | First 12 months per region are structurally null | Warmup rows dropped before modeling |

---

## Requirements

- Python 3.12+
- Internet connection (required for Step 1 only)
- ~200 MB disk space for raw + processed data and results

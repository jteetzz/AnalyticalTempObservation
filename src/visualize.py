from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT       = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data_processed"
RESULTS_DIR        = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Consistent style
plt.rcParams.update({
    "figure.dpi":      150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size":       11,
})

COLORS = {
    "co2":   "#e63946",
    "ch4":   "#f4a261",
    "n2o":   "#2a9d8f",
    "tsi":   "#457b9d",
    "aerosol": "#6d6875",
    "temp":  "#1d3557",
}


# Helpers 
def annual_mean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
# Collapse monthly data to annual means for cleaner trend lines
    return (
        df.groupby("year")[cols]
        .mean()
        .reset_index()
        .sort_values("year")
    )


def add_regression_line(ax, x, y, color="black"):
# Overlay an OLS regression line and annotate with R²
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 10:
        return
    # Cannot fit a line if all x values are identical (e.g. aerosol filled with 0)
    if np.unique(x[mask]).size < 2:
        ax.text(0.05, 0.95, "No variation in feature",
                transform=ax.transAxes, fontsize=8, color="gray",
                va="top")
        return
    slope, intercept, r, *_ = stats.linregress(x[mask], y[mask])
    x_line = np.linspace(x[mask].min(), x[mask].max(), 200)
    ax.plot(x_line, slope * x_line + intercept,
            color=color, linewidth=1.8, linestyle="--", label=f"OLS fit  R²={r**2:.3f}")
    ax.legend(fontsize=9)


# Plot 1 — GHG vs Temperature (dual axis, one subplot per gas)
def plot_ghg_vs_temp(df: pd.DataFrame) -> None:
# Three subplots side-by-side: CO2, CH4, N2O each on their own scale, with global temperature anomaly overlaid on a right-hand y-axis.
    ann = annual_mean(df, ["co2", "ch4", "n2o", "temp_anomaly"])

    gases = [
        ("co2", "CO₂ (ppm)",  COLORS["co2"]),
        ("ch4", "CH₄ (ppb)",  COLORS["ch4"]),
        ("n2o", "N₂O (ppb)",  COLORS["n2o"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle("Greenhouse Gases vs Global Temperature Anomaly", fontsize=13, fontweight="bold")

    for ax, (gas, ylabel, color) in zip(axes, gases):
        ax2 = ax.twinx()

        ax.plot(ann["year"], ann[gas], color=color, linewidth=2, label=ylabel)
        ax2.plot(ann["year"], ann["temp_anomaly"],
                 color=COLORS["temp"], linewidth=1.5, linestyle="--",
                 alpha=0.85, label="Temp anomaly (°C)")

        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel, color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax2.set_ylabel("Temp anomaly (°C)", color=COLORS["temp"])
        ax2.tick_params(axis="y", labelcolor=COLORS["temp"])
        ax.set_title(gas.upper())

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "ghg_vs_temperature.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved → {out.name}")


# Plot 2 — Natural Forcing Proxies 
def plot_natural_forcings(df: pd.DataFrame) -> None:
    #TSI and aerosol optical depth on separate axes alongside temperature
    ann = annual_mean(df, ["tsi", "aerosol_optical_depth", "temp_anomaly"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Natural Forcing Proxies vs Temperature Anomaly",
                 fontsize=13, fontweight="bold")

    # TSI
    ax1b = ax1.twinx()
    ax1.plot(ann["year"], ann["tsi"], color=COLORS["tsi"], linewidth=2, label="TSI (W/m²)")
    ax1b.plot(ann["year"], ann["temp_anomaly"],
              color=COLORS["temp"], linewidth=1.5, linestyle="--", alpha=0.8,
              label="Temp anomaly (°C)")
    ax1.set_ylabel("TSI (W/m²)", color=COLORS["tsi"])
    ax1.tick_params(axis="y", labelcolor=COLORS["tsi"])
    ax1b.set_ylabel("Temp anomaly (°C)", color=COLORS["temp"])
    ax1b.tick_params(axis="y", labelcolor=COLORS["temp"])
    ax1.set_title("Total Solar Irradiance")
    lines = ax1.get_legend_handles_labels()[0] + ax1b.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax1b.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, fontsize=9)

    # Aerosol
    ax2b = ax2.twinx()
    ax2.fill_between(ann["year"], ann["aerosol_optical_depth"],
                     color=COLORS["aerosol"], alpha=0.5, label="Aerosol optical depth")
    ax2.plot(ann["year"], ann["aerosol_optical_depth"],
             color=COLORS["aerosol"], linewidth=1.5)
    ax2b.plot(ann["year"], ann["temp_anomaly"],
              color=COLORS["temp"], linewidth=1.5, linestyle="--", alpha=0.8,
              label="Temp anomaly (°C)")
    ax2.set_ylabel("Aerosol optical depth", color=COLORS["aerosol"])
    ax2.tick_params(axis="y", labelcolor=COLORS["aerosol"])
    ax2b.set_ylabel("Temp anomaly (°C)", color=COLORS["temp"])
    ax2b.tick_params(axis="y", labelcolor=COLORS["temp"])
    ax2.set_xlabel("Year")
    ax2.set_title("Stratospheric Aerosol Optical Depth (Volcanic Proxy)")
    lines = ax2.get_legend_handles_labels()[0] + ax2b.get_legend_handles_labels()[0]
    labels = ax2.get_legend_handles_labels()[1] + ax2b.get_legend_handles_labels()[1]
    ax2.legend(lines, labels, fontsize=9)

    plt.tight_layout()
    out = RESULTS_DIR / "natural_forcings_vs_temperature.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved → {out.name}")


# Plot 3 — Regional Temperature Trends 
def plot_regional_temp(df: pd.DataFrame) -> None:
    #Annual mean temperature anomaly per region — smoothed for readability
    regions = sorted(df["region"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(regions)))

    fig, ax = plt.subplots(figsize=(13, 6))

    for region, color in zip(regions, palette):
        sub = (
            df[df["region"] == region]
            .groupby("year")["temp_anomaly"]
            .mean()
            .reset_index()
            .sort_values("year")
        )
        ax.plot(sub["year"], sub["temp_anomaly"],
                label=region, color=color, linewidth=1.6, alpha=0.85)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("Regional Temperature Anomalies (Annual Mean)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature Anomaly (°C)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout()
    out = RESULTS_DIR / "regional_temperature_trends.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved → {out.name}")


# Plot 4 — Scatter + Regression for Top Features 
def plot_scatter_top_features(df: pd.DataFrame) -> None:
    #Scatter plots with OLS fit for the most important predictors
    features = [
        ("co2",                  "CO₂ (ppm)",              COLORS["co2"]),
        ("ch4",                  "CH₄ (ppb)",              COLORS["ch4"]),
        ("tsi",                  "TSI (W/m²)",             COLORS["tsi"]),
        ("aerosol_optical_depth","Aerosol optical depth",  COLORS["aerosol"]),
    ]
    features = [(col, lbl, col_c) for col, lbl, col_c in features if col in df.columns]

    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 5))
    if len(features) == 1:
        axes = [axes]
    fig.suptitle("Feature vs Temperature Anomaly (with OLS fit)",
                 fontsize=13, fontweight="bold")

    for ax, (col, label, color) in zip(axes, features):
        x = df[col].values.astype(float)
        y = df["temp_anomaly"].values.astype(float)
        ax.scatter(x, y, alpha=0.15, s=8, color=color)
        add_regression_line(ax, x, y, color="black")
        ax.set_xlabel(label)
        ax.set_ylabel("Temperature Anomaly (°C)")
        ax.set_title(col.upper())

    plt.tight_layout()
    out = RESULTS_DIR / "scatter_top_features.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved → {out.name}")


# Plot 5 — Feature Importance Bar Charts 
def plot_feature_importance(top_n: int = 15) -> None:
    importance_files = {
        "Linear Regression\n(|Standardized Coefficient|)":
            "linear_regression_coefficients.csv",
        "Random Forest\n(Permutation Importance)":
            "RandomForest_permutation_importance.csv",
        "Gradient Boosting\n(Permutation Importance)":
            "GradientBoosting_permutation_importance.csv",
    }

    available = {
        title: RESULTS_DIR / fname
        for title, fname in importance_files.items()
        if (RESULTS_DIR / fname).exists()
    }

    if not available:
        print("  No importance CSVs found — run evaluate.py first.")
        return

    fig, axes = plt.subplots(1, len(available),
                             figsize=(8 * len(available), 7))
    if len(available) == 1:
        axes = [axes]

    fig.suptitle("Feature Importance — Root Cause Ranking",
                 fontsize=14, fontweight="bold")

    for ax, (title, fpath) in zip(axes, available.items()):
        imp = pd.read_csv(fpath, index_col=0).squeeze()
        imp = imp.abs().sort_values(ascending=False).head(top_n)
        imp = imp.sort_values()                         # ascending for barh

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(imp)))
        imp.plot(kind="barh", ax=ax, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Importance")
        ax.axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    out = RESULTS_DIR / "feature_importance_comparison.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved → {out.name}")


# Main 
def visualize() -> None:
    input_path = DATA_PROCESSED_DIR / "features_monthly_regional.csv"
    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError(f"{input_path.name} is empty.")

    df = df.sort_values(["year", "month", "region"]).reset_index(drop=True)

    print("Generating plots...")
    plot_ghg_vs_temp(df)
    plot_natural_forcings(df)
    plot_regional_temp(df)
    plot_scatter_top_features(df)
    plot_feature_importance()

    print(f"\nAll plots saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    visualize()

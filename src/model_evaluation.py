from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


PROJECT_ROOT       = Path(__file__).resolve().parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data_processed"
RESULTS_DIR        = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "temp_anomaly"

# Columns that must never be used as model inputs
DROP_COLS = {
    TARGET,
    "year", "month", "decimal_year",   # identifiers / leakage
    "temp_ma12",                        # derived from target → leakage
    "country",                          # not present but guard anyway
}


def load_and_prepare(path: Path):
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path.name} is empty.")

    # Temporal sort
    df = df.sort_values(["year", "month", "region"]).reset_index(drop=True)

    # Target
    y = df[TARGET].copy()

    # One-hot encode region for all models
    df = pd.get_dummies(df, columns=["region"], drop_first=False)

    #  Select feature columns
    # For Linear Regression: use _norm versions (standardized, comparable)
    # For tree models      : use raw versions (trees are scale-invariant)
    all_cols   = set(df.columns)
    drop_final = DROP_COLS & all_cols

    norm_cols = sorted([c for c in df.columns
                        if c.endswith("_norm") or c.startswith("region_")])
    raw_cols  = sorted([c for c in df.columns
                        if c not in drop_final
                        and not c.endswith("_norm")
                        and c != TARGET])

    X_norm = df[norm_cols].copy()
    X_raw  = df[raw_cols].copy()

    return y, X_norm, X_raw


def temporal_split(y, X_norm, X_raw, test_fraction: float = 0.30):
    n         = len(y)
    split_idx = int(n * (1 - test_fraction))

    return (
        X_norm.iloc[:split_idx], X_norm.iloc[split_idx:],
        X_raw.iloc[:split_idx],  X_raw.iloc[split_idx:],
        y.iloc[:split_idx],      y.iloc[split_idx:],
    )


def print_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    return {"model": name, "rmse": rmse, "r2": r2}


def plot_importance(series: pd.Series, title: str, save_path: Path, top_n: int = 15):
    top = series.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    top.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved plot → {save_path.name}")


def evaluate() -> None:
    input_path   = DATA_PROCESSED_DIR / "features_monthly_regional.csv"
    metrics_path = RESULTS_DIR / "model_metrics.csv"

    y, X_norm, X_raw = load_and_prepare(input_path)

    (X_norm_train, X_norm_test,
     X_raw_train,  X_raw_test,
     y_train,      y_test) = temporal_split(y, X_norm, X_raw)

    print(f"Dataset  : {len(y):,} rows")
    print(f"Train    : {len(y_train):,} rows")
    print(f"Test     : {len(y_test):,} rows")
    print(f"Lin feats: {X_norm.shape[1]}  |  Tree feats: {X_raw.shape[1]}")

    metrics_rows = []

    #  1. Linear Regression (standardized coefficients)
    lr = LinearRegression()
    lr.fit(X_norm_train, y_train)
    lr_preds = lr.predict(X_norm_test)

    metrics_rows.append(print_metrics("LinearRegression", y_test, lr_preds))

    coef = pd.Series(lr.coef_, index=X_norm.columns)
    coef_sorted = coef.reindex(coef.abs().sort_values(ascending=False).index)
    coef_sorted.to_csv(
        RESULTS_DIR / "linear_regression_coefficients.csv",
        header=["standardized_coefficient"]
    )
    plot_importance(
        coef_sorted.abs(),          # abs so bar lengths reflect magnitude
        "Linear Regression — Standardized Coefficients (Top 15)",
        RESULTS_DIR / "LinearRegression_coefficients.png",
    )

    #  2. Random Forest
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_raw_train, y_train)
    rf_preds = rf.predict(X_raw_test)

    metrics_rows.append(print_metrics("RandomForest", y_test, rf_preds))

    # Built-in impurity importance
    rf_imp = pd.Series(rf.feature_importances_, index=X_raw.columns)
    rf_imp = rf_imp.sort_values(ascending=False)
    rf_imp.to_csv(RESULTS_DIR / "RandomForest_feature_importance.csv",
                  header=["importance"])
    plot_importance(rf_imp,
                    "Random Forest — Feature Importance (Top 15)",
                    RESULTS_DIR / "RandomForest_feature_importance.png")

    # Permutation importance (more reliable, slower)
    rf_perm = permutation_importance(
        rf, X_raw_test, y_test, n_repeats=10, random_state=42, scoring="r2"
    )
    rf_perm_s = pd.Series(rf_perm.importances_mean, index=X_raw.columns)
    rf_perm_s = rf_perm_s.sort_values(ascending=False)
    rf_perm_s.to_csv(RESULTS_DIR / "RandomForest_permutation_importance.csv",
                     header=["importance"])
    plot_importance(rf_perm_s,
                    "Random Forest — Permutation Importance (Top 15)",
                    RESULTS_DIR / "RandomForest_permutation_importance.png")

    #  3. Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
    )
    gb.fit(X_raw_train, y_train)
    gb_preds = gb.predict(X_raw_test)

    metrics_rows.append(print_metrics("GradientBoosting", y_test, gb_preds))

    gb_imp = pd.Series(gb.feature_importances_, index=X_raw.columns)
    gb_imp = gb_imp.sort_values(ascending=False)
    gb_imp.to_csv(RESULTS_DIR / "GradientBoosting_feature_importance.csv",
                  header=["importance"])
    plot_importance(gb_imp,
                    "Gradient Boosting — Feature Importance (Top 15)",
                    RESULTS_DIR / "GradientBoosting_feature_importance.png")

    gb_perm = permutation_importance(
        gb, X_raw_test, y_test, n_repeats=10, random_state=42, scoring="r2"
    )
    gb_perm_s = pd.Series(gb_perm.importances_mean, index=X_raw.columns)
    gb_perm_s = gb_perm_s.sort_values(ascending=False)
    gb_perm_s.to_csv(RESULTS_DIR / "GradientBoosting_permutation_importance.csv",
                     header=["importance"])
    plot_importance(gb_perm_s,
                    "Gradient Boosting — Permutation Importance (Top 15)",
                    RESULTS_DIR / "GradientBoosting_permutation_importance.png")

    #  4. Save metrics
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\n{'='*50}")
    print("Model comparison:")
    print(metrics_df.to_string(index=False))
    print(f"\nAll results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    evaluate()

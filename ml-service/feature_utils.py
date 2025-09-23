"""
Shared feature engineering utilities for DemandAI models.

This module builds a superset of composite features used by all models
and provides helpers to select the appropriate subset per model.

Inputs expected in dataset.csv:
    - product, year, month, campaign, seasonality, quantity

Outputs:
    - A DataFrame (and optional CSV) with all composite features required
      by Bombom Lasso, Topbel Lasso, and Topbel Ridge models.

Design notes:
    - Group-wise time operations are performed per product to avoid leakage.
    - Lag/rolling features use only past information (shift/rolling with min_periods=1).
    - NaNs from past-dependent ops (first rows) are forward-filled within product;
      any remaining NaNs are set to 0 to remain conservative.
    - We keep both cyclic encodings (sin/cos) and dummies where needed.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# Catalog of features expected by each model
FEATURE_SETS: Dict[str, List[str]] = {
    # From ml-service/bombom_lasso_model.py
    "bombom_lasso": [
        "year", "month", "campaign", "seasonality", "quarter",
        "month_sin", "month_cos", "trend", "trend_squared",
        "quantity_lag1", "quantity_lag2", "quantity_lag3", "quantity_lag6",
        "quantity_ma3", "quantity_ma6", "quantity_ma12",
        "quantity_std3", "quantity_std6", "quantity_std12",
        "quantity_pct_change", "quantity_diff",
        "is_high_season", "is_low_season", "is_holiday_season",
        "campaign_season_interaction", "campaign_month", "seasonality_month",
        # Dummies
        *[f"month_{i}" for i in range(1, 13)],
        *[f"quarter_{i}" for i in range(1, 5)],
        # Optional YoY
        "quantity_yoy", "yoy_growth",
    ],

    # From ml-service/topbel_lasso_model.py
    "topbel_lasso": [
        "year", "month", "campaign", "seasonality", "quarter",
        "month_sin", "month_cos", "quarter_sin", "quarter_cos",
        "trend", "trend_squared", "trend_cubed",
        "quantity_lag1", "quantity_lag2", "quantity_lag3", "quantity_lag4", "quantity_lag6", "quantity_lag12",
        "quantity_ma3", "quantity_ma6", "quantity_ma9", "quantity_ma12",
        "quantity_std3", "quantity_std6", "quantity_std9", "quantity_std12",
        "quantity_min3", "quantity_min6", "quantity_min9", "quantity_min12",
        "quantity_max3", "quantity_max6", "quantity_max9", "quantity_max12",
        "quantity_pct_change", "quantity_diff", "quantity_diff2",
        "is_high_season", "is_low_season", "is_holiday_season", "is_summer", "is_winter",
        "campaign_season_interaction", "campaign_month", "seasonality_month",
        "campaign_high_season", "seasonality_summer",
        "quantity_volatility3", "quantity_volatility6",
        # Dummies
        *[f"month_{i}" for i in range(1, 13)],
        *[f"quarter_{i}" for i in range(1, 5)],
        # Optional YoY
        "quantity_yoy", "yoy_growth",
    ],

    # From ml-service/topbel_ridge_model.py
    "topbel_ridge": [
        "year", "month", "campaign", "seasonality", "quarter",
        "month_sin", "month_cos", "quarter_sin", "quarter_cos",
        "trend", "trend_squared",
        "quantity_lag1", "quantity_lag2", "quantity_lag3", "quantity_lag6",
        "quantity_ma3", "quantity_ma6", "quantity_ma12",
        "quantity_ewm3", "quantity_ewm6", "quantity_ewm12",
        "quantity_pct_change", "quantity_pct_change_smooth",
        "is_high_season", "is_low_season", "is_summer", "is_winter",
        "campaign_season", "campaign_high_season", "seasonality_summer",
        "quantity_stability_3", "quantity_stability_6",
        # Optional YoY
        "quantity_yoy", "yoy_growth",
    ],
}


BASE_COLUMNS = ["product", "year", "month", "campaign", "seasonality", "quantity"]


def compute_quarter(month_series: pd.Series) -> pd.Series:
    return ((month_series - 1) // 3 + 1).astype(int)


def build_features_dataset(
    input_csv: str = "dataset.csv",
    output_csv: str | None = None,
) -> pd.DataFrame:
    """Build a superset of features for all products from the raw dataset.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV with base columns (possibly raw names). Expected
        to contain a 'PESOL' column that will be dropped if present.
    output_csv : str | None
        If provided, saves the resulting DataFrame to this path.

    Returns
    -------
    pd.DataFrame
        The dataset augmented with all composite features needed by any model.
    """
    # Load raw dataset and align column names
    df_raw = pd.read_csv(input_csv, sep=",", encoding="latin1")
    if "PESOL" in df_raw.columns:
        df_raw = df_raw.drop(columns=["PESOL"])  # remove weight column

    # If the dataset isn't already standardized, rename accordingly
    if set(df_raw.columns) != set(BASE_COLUMNS):
        # Try to map common Portuguese headers to English
        rename_map = {
            "PRODUTO": "product",
            "ANO": "year",
            "MES": "month",
            "CAMPANHA": "campaign",
            "SAZONALIDADE": "seasonality",
            "QUANTIDADE": "quantity",
        }
        tmp = df_raw.rename(columns=rename_map)
        # If still missing, assume the file already uses expected names
        df = tmp
    else:
        df = df_raw

    # Ensure the critical columns exist
    missing_base = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing_base:
        raise ValueError(f"Missing required base columns in dataset: {missing_base}")

    # Keep only required bases to avoid surprises, then compute features
    df = df[BASE_COLUMNS].copy()

    # Sort by product and date
    df.sort_values(["product", "year", "month"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Temporal base features
    df["quarter"] = compute_quarter(df["month"])  # 1..4
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)

    # Trend features (per product)
    df["trend"] = df.groupby("product").cumcount().astype(float)
    df["trend_squared"] = df["trend"] ** 2
    df["trend_cubed"] = df["trend"] ** 3

    # Lag features
    for lag in [1, 2, 3, 4, 6, 12]:
        df[f"quantity_lag{lag}"] = df.groupby("product")["quantity"].shift(lag)

    # Rolling window features (per product)
    def grp_roll(series: pd.Series, window: int, func: str) -> pd.Series:
        if func == "mean":
            return series.groupby(df["product"]).transform(lambda s: s.rolling(window=window, min_periods=1).mean())
        if func == "std":
            return series.groupby(df["product"]).transform(lambda s: s.rolling(window=window, min_periods=1).std())
        if func == "min":
            return series.groupby(df["product"]).transform(lambda s: s.rolling(window=window, min_periods=1).min())
        if func == "max":
            return series.groupby(df["product"]).transform(lambda s: s.rolling(window=window, min_periods=1).max())
        raise ValueError("Unsupported rolling func")

    for window in [3, 6, 9, 12]:
        df[f"quantity_ma{window}"] = grp_roll(df["quantity"], window, "mean")
        df[f"quantity_std{window}"] = grp_roll(df["quantity"], window, "std")
        df[f"quantity_min{window}"] = grp_roll(df["quantity"], window, "min")
        df[f"quantity_max{window}"] = grp_roll(df["quantity"], window, "max")

    # Exponentially weighted means (Ridge uses these)
    for span in [3, 6, 12]:
        df[f"quantity_ewm{span}"] = (
            df.groupby("product")["quantity"].transform(lambda s: s.ewm(span=span, adjust=False).mean())
        )

    # Growth and differences (per product)
    df["quantity_pct_change"] = df.groupby("product")["quantity"].pct_change()
    df["quantity_pct_change_smooth"] = (
        df.groupby("product")["quantity_pct_change"].transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )
    df["quantity_diff"] = df.groupby("product")["quantity"].diff()
    df["quantity_diff2"] = df.groupby("product")["quantity_diff"].diff()

    # Seasonal indicators
    df["is_high_season"] = ((df["month"] >= 7) & (df["month"] <= 10)).astype(int)
    df["is_low_season"] = ((df["month"] >= 1) & (df["month"] <= 3)).astype(int)
    df["is_holiday_season"] = ((df["month"] == 12) | (df["month"] == 1)).astype(int)
    df["is_summer"] = ((df["month"] >= 6) & (df["month"] <= 8)).astype(int)
    df["is_winter"] = ((df["month"] >= 12) | (df["month"] <= 2)).astype(int)

    # Month and quarter dummies
    for m in range(1, 13):
        df[f"month_{m}"] = (df["month"] == m).astype(int)
    for q in range(1, 5):
        df[f"quarter_{q}"] = (df["quarter"] == q).astype(int)

    # Interactions
    df["campaign_season_interaction"] = df["campaign"] * df["seasonality"]
    df["campaign_season"] = df["campaign"] * df["seasonality"]  # naming for Ridge
    df["campaign_month"] = df["campaign"] * df["month"]
    df["seasonality_month"] = df["seasonality"] * df["month"]
    df["campaign_high_season"] = df["campaign"] * df["is_high_season"]
    df["seasonality_summer"] = df["seasonality"] * df["is_summer"]

    # Volatility (std / MA) with protection against div-by-zero
    for w in [3, 6]:
        std_col = f"quantity_std{w}"
        ma_col = f"quantity_ma{w}"
        vol_col = f"quantity_volatility{w}"
        denom = df[ma_col].replace(0, np.nan)
        df[vol_col] = (df[std_col] / denom).replace([np.inf, -np.inf], np.nan)

    # Stability (Ridge-specific; keep as separate names)
    df["quantity_stability_3"] = df.groupby("product")["quantity"].transform(
        lambda s: s.rolling(window=3, min_periods=1).std()
    )
    df["quantity_stability_6"] = df.groupby("product")["quantity"].transform(
        lambda s: s.rolling(window=6, min_periods=1).std()
    )

    # Year-over-year comparisons (if enough history)
    df["quantity_yoy"] = df.groupby("product")["quantity"].shift(12)
    df["yoy_growth"] = (df["quantity"] - df["quantity_yoy"]) / df["quantity_yoy"].replace(0, np.nan)

    # Non-leaking NaN handling:
    # - Forward-fill per product (uses past only), then fill remaining NaNs with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # But do not ffill the target 'quantity'
    numeric_cols_wo_target = [c for c in numeric_cols if c != "quantity"]
    df[numeric_cols_wo_target] = (
        df.groupby("product")[numeric_cols_wo_target].ffill()
    )
    # Replace inf with NaN then final fill
    df[numeric_cols_wo_target] = df[numeric_cols_wo_target].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols_wo_target] = df[numeric_cols_wo_target].fillna(0)

    # Optional save
    if output_csv:
        df.to_csv(output_csv, index=False)

    return df


def select_features(
    df: pd.DataFrame, model_key: str
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Filter a precomputed dataset to the features used by a given model.

    Returns (X_df, missing_cols, extra_cols)
    - X_df contains only columns present both in df and FEATURE_SETS[model_key].
    - missing_cols are expected by the model but absent in df.
    - extra_cols exist in df but not used by the model.
    """
    if model_key not in FEATURE_SETS:
        raise KeyError(f"Unknown model key: {model_key}. Options: {list(FEATURE_SETS)}")

    desired = FEATURE_SETS[model_key]
    present = [c for c in desired if c in df.columns]
    missing = [c for c in desired if c not in df.columns]
    extras = sorted([c for c in df.columns if c not in BASE_COLUMNS + desired])
    X = df[present].copy()
    return X, missing, extras


def load_precomputed_product(
    precomputed_csv: str,
    product_name: str,
) -> pd.DataFrame:
    """Load a precomputed features dataset and filter to a single product."""
    df = pd.read_csv(precomputed_csv)
    if "product" not in df.columns:
        raise ValueError("precomputed dataset must contain a 'product' column")
    pdf = df[df["product"] == product_name].copy()
    if pdf.empty:
        raise ValueError(f"No data found for product: {product_name}")
    pdf.sort_values(["year", "month"], inplace=True)
    pdf.reset_index(drop=True, inplace=True)
    return pdf


def _cli_build(input_csv: str, output_csv: str) -> None:
    df = build_features_dataset(input_csv=input_csv, output_csv=output_csv)
    # Simple summary
    print("\n✅ Features dataset built successfully:")
    print(f"   Input:  {input_csv}")
    print(f"   Output: {output_csv}")
    print(f"   Rows:   {len(df):,}")
    print(f"   Cols:   {len(df.columns):,}")
    # Check coverage per model
    for key, cols in FEATURE_SETS.items():
        present = sum(1 for c in cols if c in df.columns)
        print(f"   {key}: {present}/{len(cols)} features available")


def main():
    parser = argparse.ArgumentParser(description="Build a superset features dataset for all products.")
    parser.add_argument("--input", default="dataset.csv", help="Path to input dataset (default: dataset.csv)")
    parser.add_argument(
        "--output",
        default="dataset_with_features.csv",
        help="Path to output dataset (default: dataset_with_features.csv)",
    )
    args = parser.parse_args()
    _cli_build(args.input, args.output)


if __name__ == "__main__":
    main()

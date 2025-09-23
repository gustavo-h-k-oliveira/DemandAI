import os
import warnings
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _safe_cast_int(series: pd.Series) -> pd.Series:
    try:
        return series.astype(float).round().astype(int)
    except Exception:
        return pd.to_numeric(series, errors="coerce").fillna(0).round().astype(int)


def _detect_and_standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to: product, year, month, campaign, seasonality, quantity.
    The original CSV is in pt-BR and may contain encoding variations (e.g., 'Mês' -> 'M\x82s').
    We also drop 'PESOL'.
    """
    cols = list(df.columns)
    # Expect at least 7 columns in the given order per dataset.csv
    if len(cols) < 6:
        raise ValueError("CSV inesperado: menos de 6 colunas detectadas.")

    # Map by index position to be robust against accent/encoding variations
    rename_map = {
        cols[0]: "product",
        cols[1]: "year",
        cols[2]: "month",
        cols[3]: "campaign",
        cols[4]: "seasonality",
    }

    # Some files include PESOL at index 5 and QUANTIDADE at index 6
    if len(cols) >= 7:
        pesol_col = cols[5]
        qty_col = cols[6]
        rename_map[pesol_col] = "PESOL"
        rename_map[qty_col] = "quantity"
    else:
        # Fallback: try to find quantity by name heuristics
        qty_candidates = [c for c in cols if c.lower().startswith("quant")] or cols[-1:]
        rename_map[qty_candidates[0]] = "quantity"

    df = df.rename(columns=rename_map)

    # Drop obvious empty rows (many trailing commas exist in dataset.csv)
    required = ["product", "year", "month", "campaign", "seasonality", "quantity"]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # Remove PESOL if present
    if "PESOL" in df.columns:
        df = df.drop(columns=["PESOL"])  # this is not a feature we keep

    # Coerce types
    if "year" in df.columns:
        df["year"] = _safe_cast_int(df["year"])  # e.g., 2021
    if "month" in df.columns:
        df["month"] = _safe_cast_int(df["month"]).clip(lower=1, upper=12)
    if "campaign" in df.columns:
        df["campaign"] = _safe_cast_int(df["campaign"]).clip(lower=0)
    if "seasonality" in df.columns:
        df["seasonality"] = _safe_cast_int(df["seasonality"]).clip(lower=0)
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    # Ensure sorted by product, then by time
    df = df.sort_values(["product", "year", "month"]).reset_index(drop=True)
    return df


def _add_time_columns(g: pd.DataFrame) -> pd.DataFrame:
    # quarter and cyclical encodings
    dt = pd.to_datetime(g[["year", "month"]].assign(day=1), errors="coerce")
    g["quarter"] = dt.dt.quarter.astype(float)
    # Month/Quarter cyclical encodings
    g["month_sin"] = np.sin(2 * np.pi * g["month"].astype(float) / 12)
    g["month_cos"] = np.cos(2 * np.pi * g["month"].astype(float) / 12)
    g["quarter_sin"] = np.sin(2 * np.pi * g["quarter"].astype(float) / 4)
    g["quarter_cos"] = np.cos(2 * np.pi * g["quarter"].astype(float) / 4)
    return g


def _add_lags_and_rolling(g: pd.DataFrame) -> pd.DataFrame:
    # Lags (union from models)
    for lag in [1, 2, 3, 4, 6, 12]:
        g[f"quantity_lag{lag}"] = g["quantity"].shift(lag)

    # Rolling windows
    for window in [3, 6, 9, 12]:
        g[f"quantity_ma{window}"] = g["quantity"].rolling(window=window, min_periods=1).mean()
        g[f"quantity_std{window}"] = g["quantity"].rolling(window=window, min_periods=1).std()
        g[f"quantity_min{window}"] = g["quantity"].rolling(window=window, min_periods=1).min()
        g[f"quantity_max{window}"] = g["quantity"].rolling(window=window, min_periods=1).max()

    # Exponential moving averages (Ridge conservative)
    for span in [3, 6, 12]:
        g[f"quantity_ewm{span}"] = g["quantity"].ewm(span=span, adjust=False).mean()

    # Volatility measures (std/MA) for 3 and 6
    # Avoid division by zero warnings by replacing zero with NaN during division
    for window in [3, 6]:
        ma = g[f"quantity_ma{window}"].replace(0, np.nan)
        g[f"quantity_volatility{window}"] = g["quantity"].rolling(window=window, min_periods=1).std() / ma

    # Stability features (Ridge)
    g["quantity_stability_3"] = g["quantity"].rolling(window=3, min_periods=1).std()
    g["quantity_stability_6"] = g["quantity"].rolling(window=6, min_periods=1).std()
    return g


def _add_trend_and_growth(g: pd.DataFrame) -> pd.DataFrame:
    # Trend features
    g["trend"] = np.arange(len(g))
    n = max(len(g) - 1, 1)
    g["trend_norm"] = g["trend"] / n
    g["trend_squared"] = g["trend"] ** 2
    g["trend_cubed"] = g["trend"] ** 3

    # Growth/changes
    g["quantity_pct_change"] = g["quantity"].pct_change()
    g["quantity_pct_change_smooth"] = g["quantity_pct_change"].rolling(window=3, min_periods=1).mean()
    g["quantity_diff"] = g["quantity"].diff()
    g["quantity_diff2"] = g["quantity_diff"].diff()
    return g


def _add_seasonal_flags(g: pd.DataFrame) -> pd.DataFrame:
    # Seasonal flags
    m = g["month"].astype(int)
    g["is_high_season"] = ((m >= 7) & (m <= 10)).astype(int)
    g["is_low_season"] = ((m >= 1) & (m <= 3)).astype(int)
    g["is_holiday_season"] = ((m == 12) | (m == 1)).astype(int)
    g["is_summer"] = ((m >= 6) & (m <= 8)).astype(int)
    g["is_winter"] = ((m == 12) | (m <= 2)).astype(int)
    return g


def _add_yoy(g: pd.DataFrame) -> pd.DataFrame:
    g["quantity_yoy"] = g["quantity"].shift(12)
    g["yoy_diff"] = g["quantity"] - g["quantity_yoy"]
    g["yoy_growth"] = g["yoy_diff"] / g["quantity_yoy"].replace(0, np.nan)
    return g


def _add_interactions(g: pd.DataFrame) -> pd.DataFrame:
    # Basic interactions
    g["campaign_season_interaction"] = g["campaign"] * g["seasonality"]
    # Alias used in Ridge
    g["campaign_season"] = g["campaign_season_interaction"]
    g["campaign_month"] = g["campaign"] * g["month"]
    g["seasonality_month"] = g["seasonality"] * g["month"]
    g["campaign_high_season"] = g["campaign"] * g["is_high_season"]
    g["seasonality_summer"] = g["seasonality"] * g["is_summer"]
    return g


def _add_dummies(df: pd.DataFrame) -> pd.DataFrame:
    # Month and quarter dummies with fixed full set to ensure consistency
    month_dummies = pd.get_dummies(df["month"].astype(int), prefix="month")
    quarter_dummies = pd.get_dummies(df["quarter"].astype(int), prefix="quarter")

    # Ensure all expected columns exist
    for i in range(1, 13):
        col = f"month_{i}"
        if col not in month_dummies.columns:
            month_dummies[col] = 0
    for i in range(1, 5):
        col = f"quarter_{i}"
        if col not in quarter_dummies.columns:
            quarter_dummies[col] = 0

    # Order columns numerically
    month_cols = [f"month_{i}" for i in range(1, 13)]
    quarter_cols = [f"quarter_{i}" for i in range(1, 5)]
    month_dummies = month_dummies[month_cols]
    quarter_dummies = quarter_dummies[quarter_cols]

    df = pd.concat([df, month_dummies, quarter_dummies], axis=1)
    return df


def _fill_numerics(df: pd.DataFrame, group_key: str = "product") -> pd.DataFrame:
    """Fill numeric NaNs in a conservative, time-series-aware way per product.
    Strategy:
      - Replace inf with NaN
      - Per product: forward fill, then backward fill
      - Fill remaining with group mean, then with global median as last resort
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    def _fill_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g[num_cols] = g[num_cols].ffill().bfill()
        # Fill any remaining with group means
        for c in num_cols:
            if g[c].isna().any():
                g[c] = g[c].fillna(g[c].mean())
        return g

    df = df.groupby(group_key, group_keys=False).apply(_fill_group)
    # Global fallback
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


def build_features_dataset(
    input_csv: str = "dataset.csv",
    output_csv: str = "dataset_with_features.csv",
) -> str:
    """Read the base dataset and create a superset of composite features used across models.

    Returns the path to the output CSV.
    """
    if not os.path.isabs(input_csv):
        base_dir = os.path.dirname(__file__)
        input_csv = os.path.join(base_dir, input_csv)
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_csv}")

    # Read with latin1 to handle accents found in the provided CSV
    df = pd.read_csv(input_csv, sep=",", encoding="latin1")

    # Standardize columns and clean
    df = _detect_and_standardize_columns(df)

    # Add time-based columns, per product group
    def _per_product(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g = _add_time_columns(g)
        g = _add_lags_and_rolling(g)
        g = _add_trend_and_growth(g)
        g = _add_seasonal_flags(g)
        g = _add_yoy(g)
        g = _add_interactions(g)
        return g

    df = df.groupby("product", group_keys=False).apply(_per_product)

    # Month/Quarter dummies after group-calculated fields exist
    df = _add_dummies(df)

    # Fill numeric NaNs conservatively
    df = _fill_numerics(df)

    # Save output next to input
    if not os.path.isabs(output_csv):
        output_csv = os.path.join(os.path.dirname(input_csv), output_csv)

    df.to_csv(output_csv, index=False, encoding="utf-8")
    return output_csv


def main():
    print("🧮 Construindo dataset com features compostas (superset dos modelos)…")
    try:
        out_path = build_features_dataset()
        print(f"✅ Dataset com features salvo em: {out_path}")
    except Exception as e:
        print(f"❌ Falha ao construir dataset de features: {e}")


if __name__ == "__main__":
    main()

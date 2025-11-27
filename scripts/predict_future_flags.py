#!/usr/bin/env python3
"""Generate forward-looking campaign/seasonality flag predictions.

This script loads the trained flag classifiers, applies the same feature
engineering pipeline used during training, and outputs the predicted flags
(and probabilities) for the next horizon for each product.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from train_flag_classifiers import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    LABEL_NAMES,
    coerce_features,
    prepare_targets,
    stack_probabilities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict future campaign/seasonality flags")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--reference-year", type=int, default=None)
    parser.add_argument("--reference-month", type=int, default=None)
    parser.add_argument("--target-year", type=int, default=None, help="Filter predictions to a specific target year")
    parser.add_argument(
        "--target-months",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of target months (1-12) to emit",
    )
    return parser.parse_args()


def compute_target_period(year: int, month: int, horizon: int) -> Tuple[int, int]:
    total_months = year * 12 + (month - 1) + horizon
    target_year = total_months // 12
    target_month = total_months % 12 + 1
    return target_year, target_month


def select_reference_rows(
    prepared: pd.DataFrame,
    feature_columns: List[str],
    horizon: int,
    ref_year: Optional[int],
    ref_month: Optional[int],
    target_year: Optional[int],
    target_months: Optional[List[int]],
) -> pd.DataFrame:
    """Return the set of reference rows that should receive predictions."""

    if prepared.empty:
        return prepared

    if ref_year is not None and ref_month is not None:
        mask = (prepared["year"] == ref_year) & (prepared["month"] == ref_month)
        return prepared[mask].tail(1)

    feature_mask = prepared[feature_columns].notna().all(axis=1)
    candidates = prepared[feature_mask].copy()

    if candidates.empty:
        return candidates

    target_periods = candidates.apply(
        lambda row: compute_target_period(int(row["year"]), int(row["month"]), horizon), axis=1
    )
    candidates["target_year"] = [period[0] for period in target_periods]
    candidates["target_month"] = [period[1] for period in target_periods]

    if target_year is not None:
        candidates = candidates[candidates["target_year"] == target_year]

    if target_months is not None:
        valid_months = {int(month) for month in target_months}
        candidates = candidates[candidates["target_month"].isin(valid_months)]

    return candidates


def load_metadata(models_dir: Path, slug: str, horizon: int) -> Dict[str, Any]:
    meta_path = models_dir / slug / f"metrics_h{horizon}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found for {slug} at {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def collect_predictions(args: argparse.Namespace) -> Tuple[pd.DataFrame, Path]:
    summary_path = args.summary_path
    if summary_path is None:
        summary_path = args.models_dir / f"summary_h{args.horizon}.json"

    output_path = args.output_path
    if output_path is None:
        output_path = Path("data") / f"predicted_flags_h{args.horizon}.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found at {summary_path}")

    df = pd.read_csv(args.data_path)
    if "product" not in df.columns:
        raise ValueError("Dataset must contain a 'product' column")

    with open(summary_path, "r", encoding="utf-8") as fp:
        summary = json.load(fp)

    predictions: List[Dict[str, Any]] = []
    trained_models = summary.get("trained_models", [])

    for item in trained_models:
        if not item.get("trained"):
            continue

        product = item["product"]
        slug = item["slug"]

        product_df = df[df["product"] == product].copy()
        if product_df.empty:
            continue

        prepared = prepare_targets(product_df, args.horizon, drop_targets=False)
        metadata = load_metadata(args.models_dir, slug, args.horizon)
        feature_columns: List[str] = metadata["feature_columns"]
        thresholds: Dict[str, float] = metadata.get("thresholds", {label: 0.5 for label in LABEL_NAMES})
        model_path = metadata["model_path"]

        if len(feature_columns) == 0 or not Path(model_path).exists():
            continue

        reference_rows = select_reference_rows(
            prepared,
            feature_columns,
            args.horizon,
            args.reference_year,
            args.reference_month,
            args.target_year,
            args.target_months,
        )

        if reference_rows.empty:
            continue

        reference_rows = reference_rows.sort_values(["year", "month"])
        model = joblib.load(model_path)
        threshold_vector = np.array([thresholds.get(label, 0.5) for label in LABEL_NAMES], dtype=float)

        for _, reference_row in reference_rows.iterrows():
            X = coerce_features(reference_row[feature_columns].to_frame().T)
            proba = stack_probabilities(model.predict_proba(X))
            binary_preds = (proba >= threshold_vector).astype(int)

            base_year = int(reference_row["year"])
            base_month = int(reference_row["month"])
            target_year, target_month = compute_target_period(base_year, base_month, args.horizon)

            product_prediction = {
                "product": product,
                "slug": slug,
                "reference_year": base_year,
                "reference_month": base_month,
                "target_year": target_year,
                "target_month": target_month,
            }

            for idx, label in enumerate(LABEL_NAMES):
                product_prediction[f"{label}_probability"] = float(proba[0, idx])
                product_prediction[f"{label}_prediction"] = int(binary_preds[0, idx])
                product_prediction[f"{label}_threshold"] = float(threshold_vector[idx])

            predictions.append(product_prediction)

    output_df = pd.DataFrame(predictions)
    return output_df, output_path


def main() -> None:
    args = parse_args()
    output_df, output_path = collect_predictions(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Saved {len(output_df)} predictions to {output_path}")


if __name__ == "__main__":
    main()

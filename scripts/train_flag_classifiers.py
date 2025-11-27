#!/usr/bin/env python3
"""Train classifiers to predict future campaign and seasonality flags.

This script builds multi-output RandomForest classifiers (one per product)
that predict whether a product will have campaign or seasonality flags at a
future horizon (default: next month). The resulting models can later be
wired into the demand forecasting pipeline so future demand predictions can
condition on predicted flags instead of observed ones.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "dataset_with_features.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "models" / "flags"
TARGET_COLUMNS = ["campaign_target", "seasonality_target"]
LABEL_NAMES = ["campaign", "seasonality"]
EXCLUDE_COLUMNS = {
    "product",
    "quantity",
    *TARGET_COLUMNS,
}
LAG_FEATURES = {
    "quantity": [1, 2, 3],
    "campaign": [1, 2, 3],
    "seasonality": [1, 2, 3],
}
ROLLING_WINDOWS = [3, 6, 12]
PARAM_GRID = [
    {"n_estimators": 200, "max_depth": 8, "max_features": 0.6},
    {"n_estimators": 300, "max_depth": 10, "max_features": 0.5},
    {"n_estimators": 400, "max_depth": None, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": 12, "max_features": 0.7},
    {"n_estimators": 600, "max_depth": None, "max_features": 0.9},
]


@dataclass
class TrainingConfig:
    horizon: int = 1
    train_end_year: int = 2023
    min_train_rows: int = 24
    min_test_rows: int = 6
    use_cv: bool = False
    cv_splits: int = 5
    n_estimators: int = 400
    max_depth: Optional[int] = None
    max_features: Optional[Any] = "sqrt"
    random_state: int = 42
    calibrate_proba: bool = False
    calibration_method: str = "sigmoid"
    calibration_cv: int = 3
    tune_hyperparams: bool = False
    tuning_candidates: int = 4


@dataclass
class ProductResult:
    product: str
    slug: str
    trained: bool
    reason: Optional[str]
    train_rows: int
    test_rows: int
    metrics: Optional[Dict[str, Any]]
    model_path: Optional[str]
    thresholds: Optional[Dict[str, float]]
    feature_importances: Optional[Dict[str, List[Dict[str, float]]]]

def parse_max_features(value: str) -> Any:
    lowered = value.lower()
    if lowered == "auto":
        return "sqrt"
    if lowered in {"sqrt", "log2"}:
        return lowered
    try:
        return float(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train campaign/seasonality classifiers")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in months")
    parser.add_argument(
        "--train-end-year",
        type=int,
        default=2023,
        help="Last year to include in the training split (test will be > this value)",
    )
    parser.add_argument("--min-train-rows", type=int, default=24)
    parser.add_argument("--min-test-rows", type=int, default=6)
    parser.add_argument(
        "--use-cv",
        action="store_true",
        help="Use TimeSeriesSplit cross-validation instead of a single year holdout",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of TimeSeriesSplit folds when --use-cv is enabled",
    )
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--max-features", type=parse_max_features, default="sqrt")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--calibrate-proba", action="store_true", help="Wrap classifiers with CalibratedClassifierCV")
    parser.add_argument("--calibration-method", choices=["sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--calibration-cv", type=int, default=3)
    parser.add_argument("--tune-hyperparams", action="store_true", help="Run lightweight hyperparameter search per product")
    parser.add_argument("--tuning-candidates", type=int, default=4, help="How many candidate parameter sets to evaluate when tuning is enabled")
    return parser.parse_args()


def slugify(value: str) -> str:
    """Create filesystem-friendly slugs for product names."""
    return (
        value.lower()
        .replace("/", "-")
        .replace(" ", "_")
        .replace("ã", "a")
        .replace("á", "a")
        .replace("â", "a")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ô", "o")
        .replace("ú", "u")
        .replace("ç", "c")
    )


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for col, lags in LAG_FEATURES.items():
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    if "quantity" not in df.columns:
        return df
    for window in ROLLING_WINDOWS:
        df[f"quantity_roll_mean_{window}"] = df["quantity"].rolling(window, min_periods=1).mean()
        df[f"quantity_roll_std_{window}"] = df["quantity"].rolling(window, min_periods=1).std().fillna(0.0)
        df[f"quantity_roll_median_{window}"] = df["quantity"].rolling(window, min_periods=1).median()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
        df["quarter"] = ((df["month"] - 1) // 3) + 1
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    return df.fillna(0)


def prepare_targets(df: pd.DataFrame, horizon: int, drop_targets: bool = True) -> pd.DataFrame:
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df = engineer_features(df)
    df[TARGET_COLUMNS[0]] = df["campaign"].shift(-horizon)
    df[TARGET_COLUMNS[1]] = df["seasonality"].shift(-horizon)
    if drop_targets:
        df = df.dropna(subset=TARGET_COLUMNS)
    return df


def coerce_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    bool_cols = result.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        result[bool_cols] = result[bool_cols].astype(int)
    result = result.apply(pd.to_numeric, errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result


def stack_probabilities(prob_list: List[np.ndarray]) -> np.ndarray:
    columns = []
    for probs in prob_list:
        if probs.ndim == 1:
            columns.append(probs)
        else:
            columns.append(probs[:, 1] if probs.shape[1] > 1 else probs[:, 0])
    return np.column_stack(columns)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    label_names: List[str],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for idx, label in enumerate(label_names):
        true_col = y_true[:, idx]
        pred_col = y_pred[:, idx]
        proba_col = y_proba[:, idx] if y_proba is not None else None

        unique_classes = np.unique(true_col)
        metrics[label] = {
            "precision": precision_score(true_col, pred_col, zero_division=0),
            "recall": recall_score(true_col, pred_col, zero_division=0),
            "f1": f1_score(true_col, pred_col, zero_division=0),
        }
        if proba_col is not None and unique_classes.size > 1:
            try:
                metrics[label]["roc_auc"] = float(roc_auc_score(true_col, proba_col))
            except ValueError:
                metrics[label]["roc_auc"] = None
        else:
            metrics[label]["roc_auc"] = None

    metrics["joint_accuracy"] = float(np.mean((y_true == y_pred).all(axis=1)))
    return metrics


def _best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    if proba.ndim == 2:
        proba = proba[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = (2 * precision * recall) / np.clip(precision + recall, a_min=1e-9, a_max=None)
    best_idx = int(np.nanargmax(f1_scores[:-1]))  # thresholds len = len(f1)-1
    return float(thresholds[best_idx])


def compute_thresholds(y_true: np.ndarray, y_proba: np.ndarray, label_names: List[str]) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for idx, label in enumerate(label_names):
        thresholds[label] = _best_threshold(y_true[:, idx], y_proba[:, idx])
    return thresholds


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def aggregate_cv_metrics(folds: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not folds:
        return {}

    aggregate: Dict[str, Any] = {"joint_accuracy": float(np.mean([f["metrics"]["joint_accuracy"] for f in folds]))}
    for label in LABEL_NAMES:
        aggregate[label] = {}
        for metric in ["precision", "recall", "f1", "roc_auc"]:
            aggregate[label][metric] = _mean_or_none([f["metrics"][label].get(metric) for f in folds])
    return aggregate


def aggregate_thresholds(folds: List[Dict[str, Any]]) -> Dict[str, float]:
    if not folds:
        return {"campaign": 0.5, "seasonality": 0.5}
    labels = folds[0]["thresholds"].keys()
    return {
        label: float(np.mean([fold["thresholds"].get(label, 0.5) for fold in folds]))
        for label in labels
    }


def maybe_tune_hyperparams(product: str, product_df: pd.DataFrame, config: TrainingConfig) -> TrainingConfig:
    if not config.tune_hyperparams:
        return config

    if config.cv_splits < 2 or len(product_df) <= config.cv_splits:
        return config

    feature_columns = [col for col in product_df.columns if col not in EXCLUDE_COLUMNS]
    X = coerce_features(product_df[feature_columns])
    y = product_df[TARGET_COLUMNS].astype(int).values

    effective_splits = min(config.cv_splits, len(X) - 1)
    if effective_splits < 2:
        return config

    candidates = PARAM_GRID[: max(1, config.tuning_candidates)]
    best_config = config
    best_score = -np.inf

    tscv = TimeSeriesSplit(n_splits=effective_splits)

    for params in candidates:
        temp_config = replace(config, **params, tune_hyperparams=False)
        fold_scores: List[float] = []
        for train_idx, val_idx in tscv.split(X):
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            y_train_fold = y[train_idx]
            if any(np.unique(y_train_fold[:, col]).size < 2 for col in range(len(TARGET_COLUMNS))):
                continue

            model = build_classifier(temp_config)
            model.fit(X.iloc[train_idx], y_train_fold)

            y_val_pred = model.predict(X.iloc[val_idx])
            y_val_proba = stack_probabilities(model.predict_proba(X.iloc[val_idx]))
            metrics = compute_metrics(y[val_idx], y_val_pred, y_val_proba, LABEL_NAMES)
            fold_scores.append(metrics["joint_accuracy"])

        if not fold_scores:
            continue

        score = float(np.mean(fold_scores))
        if score > best_score:
            best_score = score
            best_config = temp_config

    return best_config


def extract_feature_importances(model: MultiOutputClassifier, feature_columns: List[str]) -> Dict[str, List[Dict[str, float]]]:
    importances: Dict[str, List[Dict[str, float]]] = {}
    estimators = getattr(model, "estimators_", [])
    for idx, estimator in enumerate(estimators):
        label = LABEL_NAMES[idx] if idx < len(LABEL_NAMES) else f"target_{idx}"
        if hasattr(estimator, "feature_importances_"):
            pairs = sorted(
                zip(feature_columns, estimator.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
            importances[label] = [
                {"feature": feature, "importance": float(value)}
                for feature, value in pairs[:10]
            ]
        else:
            importances[label] = []
    return importances


def summarize_hyperparams(config: TrainingConfig) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "max_features": config.max_features,
        "calibrate_proba": config.calibrate_proba,
    }
    if config.calibrate_proba:
        summary.update(
            {
                "calibration_method": config.calibration_method,
                "calibration_cv": config.calibration_cv,
            }
        )
    return summary


def build_classifier(config: TrainingConfig) -> MultiOutputClassifier:
    base_estimator = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        max_features=config.max_features,
        class_weight="balanced",
        random_state=config.random_state,
        n_jobs=-1,
    )
    estimator: Any = base_estimator
    if config.calibrate_proba:
        estimator = CalibratedClassifierCV(base_estimator, method=config.calibration_method, cv=config.calibration_cv)
    return MultiOutputClassifier(estimator)


def train_for_product(
    product: str,
    df: pd.DataFrame,
    config: TrainingConfig,
    output_dir: Path,
) -> ProductResult:
    product_df = prepare_targets(df, config.horizon)
    product_df = product_df[(product_df["campaign_target"].isin([0, 1])) & (product_df["seasonality_target"].isin([0, 1]))]

    train_df = product_df[product_df["year"] <= config.train_end_year].copy()
    test_df = product_df[product_df["year"] > config.train_end_year].copy()

    slug = slugify(product)

    min_rows_check = len(product_df if config.use_cv else train_df)
    if min_rows_check < config.min_train_rows:
        return ProductResult(product, slug, False, "Insufficient training rows", len(train_df), len(test_df), None, None, None, None)

    tuned_config = maybe_tune_hyperparams(product, product_df, config)

    if config.use_cv:
        return train_with_cv(product, slug, product_df, tuned_config, output_dir)

    if len(test_df) < config.min_test_rows:
        return ProductResult(product, slug, False, "Insufficient test rows", len(train_df), len(test_df), None, None, None, None)

    return train_with_holdout(product, slug, train_df, test_df, product_df, tuned_config, output_dir)


def train_with_holdout(
    product: str,
    slug: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    config: TrainingConfig,
    output_dir: Path,
) -> ProductResult:
    for target_col in TARGET_COLUMNS:
        if train_df[target_col].nunique() < 2:
            reason = f"Target {target_col} lacks class variability"
            return ProductResult(product, slug, False, reason, len(train_df), len(test_df), None, None, None, None)

    feature_columns = [col for col in full_df.columns if col not in EXCLUDE_COLUMNS]
    X_train = coerce_features(train_df[feature_columns])
    X_test = coerce_features(test_df[feature_columns])
    y_train = train_df[TARGET_COLUMNS].astype(int).values
    y_test = test_df[TARGET_COLUMNS].astype(int).values

    model = build_classifier(config)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob_list = model.predict_proba(X_test)
    y_proba = stack_probabilities(y_prob_list)

    metrics = compute_metrics(y_test, y_pred, y_proba, LABEL_NAMES)
    thresholds = compute_thresholds(y_test, y_proba, LABEL_NAMES)

    product_dir = output_dir / slug
    product_dir.mkdir(parents=True, exist_ok=True)
    model_path = product_dir / f"flag_classifier_h{config.horizon}.joblib"
    joblib.dump(model, model_path)

    feature_importances = extract_feature_importances(model, feature_columns)

    metadata = {
        "product": product,
        "slug": slug,
        "horizon": config.horizon,
        "train_end_year": config.train_end_year,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "feature_columns": feature_columns,
        "metrics": metrics,
        "evaluation_mode": "holdout",
        "thresholds": thresholds,
        "feature_importances": feature_importances,
        "hyperparameters": summarize_hyperparams(config),
        "model_path": str(model_path),
    }

    with open(product_dir / f"metrics_h{config.horizon}.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)

    return ProductResult(product, slug, True, None, len(train_df), len(test_df), metrics, str(model_path), thresholds, feature_importances)


def train_with_cv(
    product: str,
    slug: str,
    product_df: pd.DataFrame,
    config: TrainingConfig,
    output_dir: Path,
) -> ProductResult:
    feature_columns = [col for col in product_df.columns if col not in EXCLUDE_COLUMNS]
    X = coerce_features(product_df[feature_columns])
    y = product_df[TARGET_COLUMNS].astype(int).values

    if len(X) <= config.cv_splits:
        reason = f"Not enough rows for cv_splits={config.cv_splits}"
        return ProductResult(product, slug, False, reason, len(product_df), 0, None, None, None, None)

    folds: List[Dict[str, Any]] = []
    tscv = TimeSeriesSplit(n_splits=config.cv_splits)

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        y_train_fold = y[train_idx]
        if any(np.unique(y_train_fold[:, col]).size < 2 for col in range(len(TARGET_COLUMNS))):
            continue

        model = build_classifier(config)
        model.fit(X.iloc[train_idx], y_train_fold)

        y_val_pred = model.predict(X.iloc[val_idx])
        y_val_prob_list = model.predict_proba(X.iloc[val_idx])
        y_val_proba = stack_probabilities(y_val_prob_list)

        fold_thresholds = compute_thresholds(y[val_idx], y_val_proba, LABEL_NAMES)

        folds.append(
            {
                "fold": fold_idx,
                "train_rows": int(len(train_idx)),
                "val_rows": int(len(val_idx)),
                "metrics": compute_metrics(y[val_idx], y_val_pred, y_val_proba, LABEL_NAMES),
                "thresholds": fold_thresholds,
            }
        )

    if not folds:
        reason = "Unable to create CV folds with class variability"
        return ProductResult(product, slug, False, reason, len(product_df), 0, None, None, None, None)

    aggregate_metrics = aggregate_cv_metrics(folds)
    aggregate_thresholds_dict = aggregate_thresholds(folds)

    final_model = build_classifier(config)
    final_model.fit(X, y)

    product_dir = output_dir / slug
    product_dir.mkdir(parents=True, exist_ok=True)
    model_path = product_dir / f"flag_classifier_h{config.horizon}.joblib"
    joblib.dump(final_model, model_path)

    feature_importances = extract_feature_importances(final_model, feature_columns)

    metadata = {
        "product": product,
        "slug": slug,
        "horizon": config.horizon,
        "train_end_year": config.train_end_year,
        "train_rows": len(product_df),
        "test_rows": 0,
        "feature_columns": feature_columns,
        "metrics": aggregate_metrics,
        "evaluation_mode": "cv",
        "cv_splits": config.cv_splits,
        "cv_folds": folds,
        "thresholds": aggregate_thresholds_dict,
        "feature_importances": feature_importances,
        "hyperparameters": summarize_hyperparams(config),
        "model_path": str(model_path),
    }

    with open(product_dir / f"metrics_h{config.horizon}.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)

    return ProductResult(product, slug, True, None, len(product_df), 0, aggregate_metrics, str(model_path), aggregate_thresholds_dict, feature_importances)


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        horizon=args.horizon,
        train_end_year=args.train_end_year,
        min_train_rows=args.min_train_rows,
        min_test_rows=args.min_test_rows,
        use_cv=args.use_cv,
        cv_splits=args.cv_splits,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=args.max_features,
        random_state=args.random_state,
        calibrate_proba=args.calibrate_proba,
        calibration_method=args.calibration_method,
        calibration_cv=args.calibration_cv,
        tune_hyperparams=args.tune_hyperparams,
        tuning_candidates=args.tuning_candidates,
    )

    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {args.data_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data_path)

    if "product" not in df.columns:
        raise ValueError("Dataset must contain a 'product' column")

    product_groups = df.groupby("product")
    results: List[ProductResult] = []

    for idx, (product, product_df) in enumerate(product_groups):
        # treino para todos os produtos — sem limite de max_products
        print(f"\nTreinando modelo para: {product}")
        result = train_for_product(product, product_df, config, args.output_dir)
        if result.trained:
            print(
                "  ✔ Treinado | linhas treino: {train}, teste: {test}, joint acc: {acc:.3f}".format(
                    train=result.train_rows,
                    test=result.test_rows,
                    acc=result.metrics["joint_accuracy"] if result.metrics else float("nan"),
                )
            )
        else:
            print(f"  ✖ Pulado | Motivo: {result.reason}")
        results.append(result)

    summary = {
        "config": config.__dict__,
        "trained_models": [r.__dict__ for r in results],
    }
    summary_path = args.output_dir / f"summary_h{config.horizon}.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    trained_count = sum(1 for r in results if r.trained)
    print(
        f"\nTreinamento concluído: {trained_count} produtos treinados, "
        f"{len(results) - trained_count} pulados"
    )
    print(f"Resumo salvo em: {summary_path}")


if __name__ == "__main__":
    main()
"""CLI training orchestrator for every tabular forecasting model.

Usage:
    python -m src.models.train --model lightgbm
    python -m src.models.train --model all
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.config.settings import FORECAST, PATHS
from src.evaluation.metrics import evaluate_all
from src.models.baseline_models import get_all_baselines
from src.models.catboost_model import CatBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.utils.common import load_json, save_json, timer
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

MODEL_REGISTRY = {
    "random_forest": RandomForestModel,
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
}

_COMPARISON_REPORT_PATH = PATHS.reports_dir / "model_comparison.json"


def _load_comparison_report() -> dict:
    if _COMPARISON_REPORT_PATH.exists():
        return load_json(_COMPARISON_REPORT_PATH)
    return {}


def _update_comparison_report(model_name: str, metrics: dict, fit_seconds: float) -> None:
    report = _load_comparison_report()
    report[model_name] = {**metrics, "fit_seconds": fit_seconds}
    save_json(report, _COMPARISON_REPORT_PATH)


def train_baselines(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> None:
    """Fit and evaluate every baseline forecaster, recording results into the comparison report."""
    for baseline in get_all_baselines():
        with timer(f"baseline: {baseline.name}"):
            baseline.fit(train_df)
            preds = baseline.predict(valid_df)
        metrics = evaluate_all(valid_df[FORECAST.target_col].to_numpy(), preds)
        logger.info("%s metrics: %s", baseline.name, metrics)
        _update_comparison_report(baseline.name, metrics, fit_seconds=0.0)


def train_one_model(model_name: str) -> None:
    """Train a single registered tabular model on the persisted feature parquets."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_REGISTRY)}.")

    train = pd.read_parquet(PATHS.train_features_parquet)
    valid = pd.read_parquet(PATHS.valid_features_parquet)
    cols = load_json(PATHS.data_processed / "feature_columns.json")

    X_train, y_train = train[cols], train[FORECAST.target_col]
    X_valid, y_valid = valid[cols], valid[FORECAST.target_col]

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls()

    with timer(f"training {model_name}") as _:
        import time

        start = time.perf_counter()
        model.fit(X_train, y_train, X_valid, y_valid)
        fit_seconds = time.perf_counter() - start

    preds = model.predict(X_valid)
    metrics = evaluate_all(y_valid.to_numpy(), preds)
    logger.info("%s validation metrics: %s", model_name, metrics)

    save_path = PATHS.models_dir / f"{model_name}.joblib"
    model.save(save_path)
    logger.info("Saved %s to %s", model_name, save_path)

    _update_comparison_report(model_name, metrics, fit_seconds=fit_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tabular demand forecasting models.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=[*MODEL_REGISTRY.keys(), "all", "baselines"],
        help="Which model to train, or 'all' / 'baselines'.",
    )
    args = parser.parse_args()

    if args.model == "baselines":
        train = pd.read_parquet(PATHS.train_features_parquet)
        valid = pd.read_parquet(PATHS.valid_features_parquet)
        train_baselines(train, valid)
    elif args.model == "all":
        for name in MODEL_REGISTRY:
            train_one_model(name)
    else:
        train_one_model(args.model)


if __name__ == "__main__":
    main()

"""MLflow experiment tracking helpers.

Wraps model training in an MLflow run that logs hyperparameters, metrics,
the trained model artifact (via joblib), and any accompanying plots, so
every experiment is reproducible and comparable from the MLflow UI
(``mlflow ui --backend-store-uri file:./mlruns``).
"""

from __future__ import annotations

import mlflow
import pandas as pd

from src.config.settings import FORECAST, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, PATHS
from src.evaluation.metrics import evaluate_all
from src.models.train import MODEL_REGISTRY
from src.utils.common import load_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _configure_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def train_and_log_model(model_name: str, params_override: dict | None = None) -> dict:
    """Train one registered model inside an MLflow run and log everything.

    Args:
        model_name: Key into ``src.models.train.MODEL_REGISTRY``.
        params_override: Optional hyperparameter overrides (e.g. from an
            Optuna study's ``best_params``).

    Returns:
        The validation metrics dict for the run.
    """
    _configure_mlflow()

    train = pd.read_parquet(PATHS.train_features_parquet)
    valid = pd.read_parquet(PATHS.valid_features_parquet)
    feature_columns = load_json(PATHS.data_processed / "feature_columns.json")

    X_train, y_train = train[feature_columns], train[FORECAST.target_col]
    X_valid, y_valid = valid[feature_columns], valid[FORECAST.target_col]

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(**(params_override or {}))

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("n_train_rows", len(train))
        mlflow.log_param("n_valid_rows", len(valid))
        mlflow.log_param("horizon_days", FORECAST.horizon_days)
        for k, v in (params_override or {}).items():
            mlflow.log_param(f"hp_{k}", v)

        import time

        start = time.perf_counter()
        model.fit(X_train, y_train, X_valid, y_valid)
        fit_seconds = time.perf_counter() - start

        preds = model.predict(X_valid)
        metrics = evaluate_all(y_valid.to_numpy(), preds)

        mlflow.log_metric("fit_seconds", fit_seconds)
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        save_path = PATHS.models_dir / f"{model_name}_mlflow.joblib"
        model.save(save_path)
        mlflow.log_artifact(str(save_path), artifact_path="model")

        logger.info("Logged MLflow run for %s: %s", model_name, metrics)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lightgbm", choices=list(MODEL_REGISTRY))
    args = parser.parse_args()
    train_and_log_model(args.model)

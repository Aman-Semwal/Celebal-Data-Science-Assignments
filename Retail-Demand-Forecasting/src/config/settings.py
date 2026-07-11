"""Central configuration for the Retail Demand Forecasting project.

All paths, model hyperparameter defaults, and pipeline constants are defined
here so that every module (data, features, models, api, dashboard) shares a
single source of truth.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is three levels above this file: src/config/settings.py -> repo root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Paths(BaseSettings):
    """Filesystem locations used throughout the pipeline."""

    root: Path = PROJECT_ROOT
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"
    reports_dir: Path = PROJECT_ROOT / "reports"
    images_dir: Path = PROJECT_ROOT / "images"

    train_csv: Path = PROJECT_ROOT / "data" / "raw" / "train.csv"
    test_csv: Path = PROJECT_ROOT / "data" / "raw" / "test.csv"

    processed_features_parquet: Path = PROJECT_ROOT / "data" / "processed" / "features.parquet"
    train_features_parquet: Path = PROJECT_ROOT / "data" / "processed" / "train_features.parquet"
    valid_features_parquet: Path = PROJECT_ROOT / "data" / "processed" / "valid_features.parquet"
    test_features_parquet: Path = PROJECT_ROOT / "data" / "processed" / "test_features.parquet"

    def ensure_dirs(self) -> None:
        """Create all output directories if they do not already exist."""
        for path in (
            self.data_raw,
            self.data_processed,
            self.models_dir,
            self.reports_dir,
            self.images_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


class ForecastConfig(BaseSettings):
    """Core forecasting parameters."""

    model_config = SettingsConfigDict(env_prefix="FORECAST_")

    horizon_days: int = 90
    date_col: str = "date"
    target_col: str = "sales"
    store_col: str = "store"
    item_col: str = "item"

    # Validation window: last N days of the training history held out
    # for time-based validation (must be >= horizon_days).
    validation_days: int = 90

    # Lag / rolling-window features (in days)
    lag_days: list[int] = Field(default_factory=lambda: [1, 7, 14, 28, 90, 365])
    rolling_windows: list[int] = Field(default_factory=lambda: [7, 14, 28, 90])

    # Number of splits for TimeSeriesSplit cross-validation
    n_cv_splits: int = 5

    random_seed: int = 42


class ModelDefaults(BaseSettings):
    """Default hyperparameters for each model family (pre-tuning)."""

    random_forest: dict = Field(
        default_factory=lambda: {
            "n_estimators": 60,
            "max_depth": 10,
            "min_samples_leaf": 20,
            "max_samples": 0.25,
            "n_jobs": -1,
            "random_state": 42,
        }
    )
    lightgbm: dict = Field(
        default_factory=lambda: {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
    )
    xgboost: dict = Field(
        default_factory=lambda: {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }
    )
    catboost: dict = Field(
        default_factory=lambda: {
            "iterations": 800,
            "learning_rate": 0.03,
            "depth": 8,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "verbose": False,
        }
    )
    attention_lstm: dict = Field(
        default_factory=lambda: {
            "sequence_length": 90,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "batch_size": 256,
            "epochs": 15,
        }
    )


PATHS = Paths()
FORECAST = ForecastConfig()
MODEL_DEFAULTS = ModelDefaults()

MLFLOW_TRACKING_URI: str = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME: str = "retail-demand-forecasting"

FEATURE_COLUMNS_CACHE: Path = PATHS.data_processed / "feature_columns.json"

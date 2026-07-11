"""Inference service used by both the FastAPI app and the Streamlit dashboard.

Loads the best trained model plus the full daily sales history once, and
builds the same horizon-safe feature row (calendar + lag/rolling +
aggregates) for any (store, item, date) request using the identical
feature-engineering functions used at training time — so serving-time
features can never silently drift from training-time features.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from src.config.settings import FORECAST, PATHS
from src.data.ingestion import load_train
from src.data.preprocessing import fill_calendar_gaps
from src.features.pipeline import build_feature_table
from src.utils.common import load_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Preference order: try the best-performing model first, fall back if missing.
_MODEL_PREFERENCE = ["catboost", "xgboost", "lightgbm_tuned", "lightgbm", "random_forest"]


class InferenceService:
    """Loads a trained model + history once, and serves single/multi-day forecasts."""

    def __init__(self, model_name: str | None = None) -> None:
        self.history: pd.DataFrame = fill_calendar_gaps(load_train())
        self.feature_columns: list[str] = load_json(PATHS.data_processed / "feature_columns.json")
        self.model_name, self.model = self._load_model(model_name)
        self.validation_metrics: dict = self._load_validation_metrics(self.model_name)

    def _load_model(self, model_name: str | None):
        candidates = [model_name] if model_name else _MODEL_PREFERENCE
        for name in candidates:
            if name is None:
                continue
            path = PATHS.models_dir / f"{name}.joblib"
            if path.exists():
                payload = joblib.load(path)

                # Reconstruct the correct wrapper class based on the stored name,
                # then overwrite its default-initialized model/features with the
                # actual trained artifact.
                wrapper = _wrapper_for(payload["name"])
                wrapper.model = payload["model"]
                wrapper.feature_columns = payload["feature_columns"]
                logger.info("Loaded model '%s' from %s", name, path)
                return name, wrapper
        raise FileNotFoundError(
            f"No trained model found in {PATHS.models_dir}. Train at least one model first "
            f"(e.g. `python -m src.models.train --model catboost`)."
        )

    @staticmethod
    def _load_validation_metrics(model_name: str) -> dict:
        report_path = PATHS.reports_dir / "model_comparison.json"
        if not report_path.exists():
            return {}
        report = load_json(report_path)
        return report.get(model_name, {})

    def _build_request_row(self, store: int, item: int, target_date) -> pd.DataFrame:
        """Append one (store, item, target_date) row to history and featurize it."""
        target_date = pd.Timestamp(target_date)
        scaffold = pd.DataFrame(
            {
                FORECAST.date_col: [target_date],
                FORECAST.store_col: [store],
                FORECAST.item_col: [item],
                FORECAST.target_col: [pd.NA],
            }
        )
        series_history = self.history[
            (self.history[FORECAST.store_col] == store) & (self.history[FORECAST.item_col] == item)
        ]
        if series_history.empty:
            raise ValueError(f"No history found for store={store}, item={item}.")

        combined = (
            pd.concat(
                [
                    series_history[
                        [
                            FORECAST.date_col,
                            FORECAST.store_col,
                            FORECAST.item_col,
                            FORECAST.target_col,
                        ]
                    ],
                    scaffold,
                ],
                ignore_index=True,
            )
            .sort_values(FORECAST.date_col)
            .reset_index(drop=True)
        )

        featured = build_feature_table(combined, reference_df=series_history)
        request_row = featured[featured[FORECAST.date_col] == target_date]
        return request_row

    def predict_single(self, store: int, item: int, target_date) -> float:
        """Predict sales for one (store, item, date)."""
        row = self._build_request_row(store, item, target_date)
        preds = self.model.predict(row[self.feature_columns])
        return float(preds[0])

    def predict_horizon(self, store: int, item: int, horizon_days: int = 90) -> pd.DataFrame:
        """Predict sales for the next ``horizon_days`` days starting the day after the last known history."""
        last_date = self.history[FORECAST.date_col].max()
        future_dates = pd.date_range(
            last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D"
        )

        scaffold = pd.DataFrame(
            {
                FORECAST.date_col: future_dates,
                FORECAST.store_col: store,
                FORECAST.item_col: item,
                FORECAST.target_col: pd.NA,
            }
        )
        series_history = self.history[
            (self.history[FORECAST.store_col] == store) & (self.history[FORECAST.item_col] == item)
        ]
        if series_history.empty:
            raise ValueError(f"No history found for store={store}, item={item}.")

        combined = (
            pd.concat(
                [
                    series_history[
                        [
                            FORECAST.date_col,
                            FORECAST.store_col,
                            FORECAST.item_col,
                            FORECAST.target_col,
                        ]
                    ],
                    scaffold,
                ],
                ignore_index=True,
            )
            .sort_values(FORECAST.date_col)
            .reset_index(drop=True)
        )

        featured = build_feature_table(combined, reference_df=series_history)
        future_rows = featured[featured[FORECAST.date_col].isin(future_dates)].sort_values(
            FORECAST.date_col
        )

        preds = self.model.predict(future_rows[self.feature_columns])
        return pd.DataFrame(
            {FORECAST.date_col: future_rows[FORECAST.date_col].values, "predicted_sales": preds}
        )


def _wrapper_for(name: str):
    """Instantiate the correct model wrapper class for a saved model name."""
    from src.models.catboost_model import CatBoostModel
    from src.models.lightgbm_model import LightGBMModel
    from src.models.random_forest_model import RandomForestModel
    from src.models.xgboost_model import XGBoostModel

    registry = {
        "catboost": CatBoostModel,
        "lightgbm": LightGBMModel,
        "xgboost": XGBoostModel,
        "random_forest": RandomForestModel,
    }
    cls = registry.get(name, LightGBMModel)
    return cls()


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Return a process-wide singleton ``InferenceService`` (loaded once, reused across requests)."""
    return InferenceService()

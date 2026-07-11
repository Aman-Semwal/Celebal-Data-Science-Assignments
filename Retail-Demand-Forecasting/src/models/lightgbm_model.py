"""LightGBM demand forecaster with native categorical feature support."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from src.config.settings import FORECAST, MODEL_DEFAULTS
from src.models.base_model import BaseForecastModel


class LightGBMModel(BaseForecastModel):
    """Gradient-boosted trees via LightGBM, using native categorical handling for store/item."""

    name = "lightgbm"

    def __init__(self, **override_params) -> None:
        super().__init__()
        params = {**MODEL_DEFAULTS.lightgbm, **override_params}
        self.model = LGBMRegressor(**params)
        self._cat_features = [FORECAST.store_col, FORECAST.item_col]

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> LightGBMModel:
        self.feature_columns = list(X_train.columns)
        cat_features = [c for c in self._cat_features if c in X_train.columns]

        callbacks = [log_evaluation(period=0)]
        eval_set = None
        if X_valid is not None and y_valid is not None:
            eval_set = [(X_valid[self.feature_columns], y_valid)]
            callbacks.append(early_stopping(stopping_rounds=50, verbose=False))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            categorical_feature=cat_features,
            callbacks=callbacks,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_sel = self._select_features(X)
        preds = self.model.predict(X_sel)
        return np.clip(preds, 0, None)

    def get_feature_importance(self) -> pd.Series:
        """Return gain-based feature importances, sorted descending."""
        importances = pd.Series(self.model.feature_importances_, index=self.feature_columns)
        return importances.sort_values(ascending=False)

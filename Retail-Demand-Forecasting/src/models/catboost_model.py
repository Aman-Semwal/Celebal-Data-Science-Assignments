"""CatBoost demand forecaster using native categorical feature support."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.config.settings import FORECAST, MODEL_DEFAULTS
from src.models.base_model import BaseForecastModel


class CatBoostModel(BaseForecastModel):
    """Gradient-boosted trees via CatBoost, with native categorical handling for store/item."""

    name = "catboost"

    def __init__(self, **override_params) -> None:
        super().__init__()
        params = {**MODEL_DEFAULTS.catboost, **override_params}
        self.model = CatBoostRegressor(**params)
        self._cat_features = [FORECAST.store_col, FORECAST.item_col]

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> CatBoostModel:
        self.feature_columns = list(X_train.columns)
        cat_features = [c for c in self._cat_features if c in X_train.columns]

        # CatBoost categorical columns must not contain NaN and work best as strings.
        X_train_cb = X_train.copy()
        for c in cat_features:
            X_train_cb[c] = X_train_cb[c].astype(str)
        train_pool = Pool(X_train_cb, y_train, cat_features=cat_features)

        eval_pool = None
        if X_valid is not None and y_valid is not None:
            X_valid_cb = X_valid[self.feature_columns].copy()
            for c in cat_features:
                X_valid_cb[c] = X_valid_cb[c].astype(str)
            eval_pool = Pool(X_valid_cb, y_valid, cat_features=cat_features)

        self.model.fit(
            train_pool,
            eval_set=eval_pool,
            use_best_model=eval_pool is not None,
            early_stopping_rounds=50 if eval_pool else None,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_sel = self._select_features(X).copy()
        for c in self._cat_features:
            if c in X_sel.columns:
                X_sel[c] = X_sel[c].astype(str)
        preds = self.model.predict(X_sel)
        return np.clip(preds, 0, None)

    def get_feature_importance(self) -> pd.Series:
        """Return CatBoost's built-in feature importances, sorted descending."""
        importances = pd.Series(self.model.get_feature_importance(), index=self.feature_columns)
        return importances.sort_values(ascending=False)

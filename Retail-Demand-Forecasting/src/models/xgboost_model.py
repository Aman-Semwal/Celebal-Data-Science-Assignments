"""XGBoost demand forecaster using native categorical support (enable_categorical)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.config.settings import MODEL_DEFAULTS
from src.models.base_model import BaseForecastModel


class XGBoostModel(BaseForecastModel):
    """Gradient-boosted trees via XGBoost, with native pandas categorical support."""

    name = "xgboost"

    def __init__(self, **override_params) -> None:
        super().__init__()
        params = {**MODEL_DEFAULTS.xgboost, **override_params, "enable_categorical": True}
        self.model = XGBRegressor(**params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> XGBoostModel:
        self.feature_columns = list(X_train.columns)

        fit_kwargs: dict = {}
        if X_valid is not None and y_valid is not None:
            fit_kwargs["eval_set"] = [(X_valid[self.feature_columns], y_valid)]
            fit_kwargs["verbose"] = False

        self.model.fit(X_train, y_train, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_sel = self._select_features(X)
        preds = self.model.predict(X_sel)
        return np.clip(preds, 0, None)

    def get_feature_importance(self) -> pd.Series:
        """Return gain-based feature importances, sorted descending."""
        importances = pd.Series(self.model.feature_importances_, index=self.feature_columns)
        return importances.sort_values(ascending=False)

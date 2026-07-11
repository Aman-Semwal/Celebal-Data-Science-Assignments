"""Random Forest demand forecaster.

Categorical columns (store, item) are one-hot/ordinal-encoded since
scikit-learn's RandomForestRegressor does not natively support pandas
categoricals the way the gradient boosting libraries do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config.settings import MODEL_DEFAULTS
from src.models.base_model import BaseForecastModel


class RandomForestModel(BaseForecastModel):
    """Random Forest regressor over engineered tabular features."""

    name = "random_forest"

    def __init__(self, **override_params) -> None:
        super().__init__()
        params = {**MODEL_DEFAULTS.random_forest, **override_params}
        self.model = RandomForestRegressor(**params)

    @staticmethod
    def _encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.select_dtypes(include="category").columns:
            X[col] = X[col].astype("int32")
        return X.fillna(-1)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> RandomForestModel:
        self.feature_columns = list(X_train.columns)
        X_enc = self._encode_categoricals(X_train)
        self.model.fit(X_enc, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_sel = self._select_features(X)
        X_enc = self._encode_categoricals(X_sel)
        preds = self.model.predict(X_enc)
        return np.clip(preds, 0, None)

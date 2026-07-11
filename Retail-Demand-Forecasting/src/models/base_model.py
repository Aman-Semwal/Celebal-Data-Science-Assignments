"""Common interface for every ML forecasting model in the project.

Every model (RandomForest, LightGBM, XGBoost, CatBoost, Attention-LSTM)
implements this interface so the training script, evaluation, SHAP, and
API/dashboard layers can treat them interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class BaseForecastModel(ABC):
    """Abstract interface for a tabular Store x Item demand forecasting model."""

    name: str = "base_model"

    def __init__(self) -> None:
        self.model: object | None = None
        self.feature_columns: list[str] = []

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> BaseForecastModel:
        """Fit the model on training data, optionally using a validation set for early stopping."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict sales for each row of ``X``. Predictions are clipped to be non-negative."""

    def save(self, path: Path) -> None:
        """Persist the fitted model (and feature column list) to disk via joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "feature_columns": self.feature_columns, "name": self.name}, path
        )

    def load(self, path: Path) -> BaseForecastModel:
        """Load a previously saved model from disk."""
        payload = joblib.load(path)
        self.model = payload["model"]
        self.feature_columns = payload["feature_columns"]
        self.name = payload["name"]
        return self

    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only the columns this model was trained on, in the correct order."""
        if not self.feature_columns:
            return X
        return X[self.feature_columns]

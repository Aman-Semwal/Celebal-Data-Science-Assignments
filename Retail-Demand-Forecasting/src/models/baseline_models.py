"""Simple baseline forecasters used as a floor for model comparison.

All baselines operate per (store, item) series and predict a constant (or
day-of-week-varying) value across the entire 90-day horizon, using only
information available at the as-of date (``target_date - horizon_days``),
consistent with the leakage-safe design used for the tree models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.config.settings import FORECAST


class BaselineForecaster(ABC):
    """Common interface for all baseline forecasters."""

    name: str = "baseline"

    @abstractmethod
    def fit(self, history_df: pd.DataFrame) -> BaselineForecaster:
        """Fit the baseline on historical (date, store, item, sales) data."""

    @abstractmethod
    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        """Predict sales for each row of ``future_df`` (needs date/store/item)."""


class NaiveLastValueForecaster(BaselineForecaster):
    """Predicts the last known value (as of ``target_date - horizon_days``) for every future day."""

    name = "naive_last_value"

    def __init__(self, horizon_days: int | None = None) -> None:
        self.horizon_days = horizon_days or FORECAST.horizon_days
        self._last_values: pd.Series | None = None

    def fit(self, history_df: pd.DataFrame) -> NaiveLastValueForecaster:
        idx = [FORECAST.store_col, FORECAST.item_col]
        last_date = history_df[FORECAST.date_col].max()
        last_rows = history_df[history_df[FORECAST.date_col] == last_date]
        self._last_values = last_rows.set_index(idx)[FORECAST.target_col]
        return self

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        if self._last_values is None:
            raise RuntimeError("Call fit() before predict().")
        idx = [FORECAST.store_col, FORECAST.item_col]
        keys = pd.MultiIndex.from_frame(future_df[idx])
        return self._last_values.reindex(keys).fillna(self._last_values.mean()).to_numpy()


class SeasonalNaiveForecaster(BaselineForecaster):
    """Predicts the value observed exactly ``season_length`` days before the target date."""

    name = "seasonal_naive"

    def __init__(self, season_length: int = 365) -> None:
        self.season_length = season_length
        self._history: pd.Series | None = None

    def fit(self, history_df: pd.DataFrame) -> SeasonalNaiveForecaster:
        idx = [FORECAST.store_col, FORECAST.item_col, FORECAST.date_col]
        self._history = history_df.set_index(idx)[FORECAST.target_col].sort_index()
        return self

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        if self._history is None:
            raise RuntimeError("Call fit() before predict().")
        preds = []
        fallback_mean = self._history.mean()
        for _, row in future_df.iterrows():
            lookup_date = row[FORECAST.date_col] - pd.Timedelta(days=self.season_length)
            key = (row[FORECAST.store_col], row[FORECAST.item_col], lookup_date)
            preds.append(self._history.get(key, fallback_mean))
        return np.array(preds, dtype="float32")


class MovingAverageForecaster(BaselineForecaster):
    """Predicts the mean of the last ``window`` days ending at the as-of date."""

    name = "moving_average"

    def __init__(self, window: int = 28, horizon_days: int | None = None) -> None:
        self.window = window
        self.horizon_days = horizon_days or FORECAST.horizon_days
        self._series_means: pd.Series | None = None

    def fit(self, history_df: pd.DataFrame) -> MovingAverageForecaster:
        idx = [FORECAST.store_col, FORECAST.item_col]
        cutoff = history_df[FORECAST.date_col].max() - pd.Timedelta(days=self.window - 1)
        recent = history_df[history_df[FORECAST.date_col] >= cutoff]
        self._series_means = recent.groupby(idx)[FORECAST.target_col].mean()
        return self

    def predict(self, future_df: pd.DataFrame) -> np.ndarray:
        if self._series_means is None:
            raise RuntimeError("Call fit() before predict().")
        idx = [FORECAST.store_col, FORECAST.item_col]
        keys = pd.MultiIndex.from_frame(future_df[idx])
        return self._series_means.reindex(keys).fillna(self._series_means.mean()).to_numpy()


def get_all_baselines() -> list[BaselineForecaster]:
    """Return one instance of every baseline forecaster."""
    return [
        NaiveLastValueForecaster(),
        SeasonalNaiveForecaster(season_length=365),
        MovingAverageForecaster(window=28),
    ]

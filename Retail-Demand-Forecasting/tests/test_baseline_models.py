"""Tests for src.models.baseline_models."""

from __future__ import annotations

import numpy as np
from src.models.baseline_models import (
    MovingAverageForecaster,
    NaiveLastValueForecaster,
    SeasonalNaiveForecaster,
    get_all_baselines,
)


def test_get_all_baselines_returns_three_models():
    baselines = get_all_baselines()
    assert len(baselines) == 3
    assert {b.name for b in baselines} == {"naive_last_value", "seasonal_naive", "moving_average"}


def test_naive_last_value_predicts_constant_per_series(synthetic_train_df, synthetic_test_df):
    model = NaiveLastValueForecaster().fit(synthetic_train_df)
    preds = model.predict(synthetic_test_df)
    assert len(preds) == len(synthetic_test_df)
    assert np.all(np.isfinite(preds))

    subset_mask = (synthetic_test_df["store"] == 1) & (synthetic_test_df["item"] == 1)
    subset_preds = preds[subset_mask.to_numpy()]
    assert len(set(subset_preds)) == 1


def test_moving_average_predicts_reasonable_range(synthetic_train_df, synthetic_test_df):
    model = MovingAverageForecaster(window=28).fit(synthetic_train_df)
    preds = model.predict(synthetic_test_df)
    assert np.all(preds >= 0)
    assert np.all(np.isfinite(preds))


def test_seasonal_naive_falls_back_gracefully(synthetic_train_df, synthetic_test_df):
    model = SeasonalNaiveForecaster(season_length=365).fit(synthetic_train_df)
    preds = model.predict(synthetic_test_df)
    assert len(preds) == len(synthetic_test_df)
    assert np.all(np.isfinite(preds))

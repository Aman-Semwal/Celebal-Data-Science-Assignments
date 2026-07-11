"""Tests for src.features.calendar_features and src.features.lag_features."""

from __future__ import annotations

import pandas as pd
from src.features.calendar_features import add_calendar_features
from src.features.lag_features import add_lag_and_rolling_features


def test_calendar_features_adds_expected_columns(synthetic_train_df):
    out = add_calendar_features(synthetic_train_df)
    expected = {
        "year",
        "month",
        "day",
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "quarter",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
    }
    assert expected.issubset(out.columns)


def test_calendar_features_weekend_flag_correct(synthetic_train_df):
    out = add_calendar_features(synthetic_train_df)
    saturday_rows = out[out["date"].dt.day_name() == "Saturday"]
    assert (saturday_rows["is_weekend"] == 1).all()
    monday_rows = out[out["date"].dt.day_name() == "Monday"]
    assert (monday_rows["is_weekend"] == 0).all()


def test_lag_features_are_horizon_shifted_correctly():
    """The lag_1 feature at row i must equal sales exactly (horizon_days + 1) days earlier."""
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    df = pd.DataFrame({"date": dates, "store": 1, "item": 1, "sales": range(200)})

    out = add_lag_and_rolling_features(df, lag_days=[1], rolling_windows=[7], horizon_days=90)

    row_150 = out[out["date"] == dates[150]].iloc[0]
    expected_lag1_date_idx = 150 - 90 - 1
    assert row_150["lag_1"] == df.iloc[expected_lag1_date_idx]["sales"]


def test_lag_features_no_leakage_within_horizon():
    """Rows within the first `horizon_days` of history must have NaN lag features (no leakage possible)."""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    df = pd.DataFrame({"date": dates, "store": 1, "item": 1, "sales": range(50)})

    out = add_lag_and_rolling_features(df, lag_days=[1], rolling_windows=[7], horizon_days=90)
    assert (
        out["lag_1"].isna().all()
    )  # not enough history for any row to have a valid horizon-shifted lag


def test_lag_features_grouped_by_series(synthetic_train_df):
    out = add_lag_and_rolling_features(
        synthetic_train_df, lag_days=[1], rolling_windows=[7], horizon_days=90
    )
    series_a = out[(out["store"] == 1) & (out["item"] == 1)].sort_values("date")
    series_b = out[(out["store"] == 2) & (out["item"] == 2)].sort_values("date")
    assert (
        not series_a["lag_1"]
        .reset_index(drop=True)
        .equals(series_b["lag_1"].reset_index(drop=True))
    )

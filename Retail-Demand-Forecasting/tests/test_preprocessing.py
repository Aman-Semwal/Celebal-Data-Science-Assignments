"""Tests for src.data.preprocessing."""

from __future__ import annotations

import pandas as pd
from src.data.preprocessing import fill_calendar_gaps, time_based_train_valid_split


def test_fill_calendar_gaps_is_noop_on_continuous_data(synthetic_train_df):
    filled = fill_calendar_gaps(synthetic_train_df)
    assert len(filled) == len(synthetic_train_df)


def test_fill_calendar_gaps_fills_missing_days():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03")],
            "store": [1, 1],
            "item": [1, 1],
            "sales": [5, 7],
        }
    )
    filled = fill_calendar_gaps(df)
    assert len(filled) == 3  # Jan 1, 2, 3
    middle_row = filled[filled["date"] == pd.Timestamp("2020-01-02")]
    assert middle_row["sales"].iloc[0] == 0


def test_time_based_split_respects_validation_window(synthetic_train_df):
    train, valid = time_based_train_valid_split(synthetic_train_df, validation_days=90)
    assert train["date"].max() < valid["date"].min()
    assert (valid["date"].max() - valid["date"].min()).days == 89


def test_time_based_split_covers_all_rows(synthetic_train_df):
    train, valid = time_based_train_valid_split(synthetic_train_df, validation_days=90)
    assert len(train) + len(valid) == len(synthetic_train_df)

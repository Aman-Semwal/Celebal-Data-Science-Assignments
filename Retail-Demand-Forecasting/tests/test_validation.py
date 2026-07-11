"""Tests for src.data.validation."""

from __future__ import annotations

import pandas as pd
import pytest
from src.data.validation import (
    DataValidationError,
    validate_test,
    validate_train,
    validate_train_test_consistency,
)


def test_validate_train_passes_on_clean_data(synthetic_train_df):
    validate_train(synthetic_train_df)  # should not raise


def test_validate_train_rejects_negative_sales(synthetic_train_df):
    bad_df = synthetic_train_df.copy()
    bad_df.loc[0, "sales"] = -5
    with pytest.raises(DataValidationError, match="Negative sales"):
        validate_train(bad_df)


def test_validate_train_rejects_missing_columns():
    bad_df = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "store": [1]})
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_train(bad_df)


def test_validate_train_rejects_duplicates(synthetic_train_df):
    bad_df = pd.concat([synthetic_train_df, synthetic_train_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_train(bad_df)


def test_validate_test_passes_on_clean_data(synthetic_test_df):
    validate_test(synthetic_test_df)  # should not raise


def test_validate_train_test_consistency_passes(synthetic_train_df, synthetic_test_df):
    validate_train_test_consistency(synthetic_train_df, synthetic_test_df)  # should not raise


def test_validate_train_test_consistency_rejects_unseen_series(
    synthetic_train_df, synthetic_test_df
):
    bad_test = synthetic_test_df.copy()
    bad_test.loc[0, "item"] = 999
    with pytest.raises(DataValidationError, match="no training history"):
        validate_train_test_consistency(synthetic_train_df, bad_test)


def test_validate_train_test_consistency_rejects_overlap(synthetic_train_df):
    overlapping_test = synthetic_train_df.rename(columns={}).copy()
    overlapping_test["id"] = range(len(overlapping_test))
    with pytest.raises(DataValidationError, match="overlaps"):
        validate_train_test_consistency(synthetic_train_df, overlapping_test)

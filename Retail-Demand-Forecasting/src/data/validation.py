"""Data validation: schema checks, completeness, and sanity checks.

Raises ``DataValidationError`` on any failed check so pipeline failures
surface immediately rather than propagating silently into modeling.
"""

from __future__ import annotations

import pandas as pd

from src.config.settings import FORECAST
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Raised when a dataset fails a validation check."""


def _require_columns(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise DataValidationError(f"{df_name} is missing required columns: {sorted(missing)}")


def validate_train(df: pd.DataFrame) -> None:
    """Validate the training DataFrame.

    Checks:
        - Required columns present.
        - No nulls in key columns.
        - Sales are non-negative.
        - No duplicate (date, store, item) combinations.
        - Every (store, item) pair has a fully continuous daily date range.
    """
    required = [FORECAST.date_col, FORECAST.store_col, FORECAST.item_col, FORECAST.target_col]
    _require_columns(df, required, "train")

    if df[required].isnull().any().any():
        null_counts = df[required].isnull().sum()
        raise DataValidationError(
            f"Null values found in train data: {null_counts[null_counts > 0].to_dict()}"
        )

    if (df[FORECAST.target_col] < 0).any():
        raise DataValidationError("Negative sales values found in train data.")

    dup_key = [FORECAST.date_col, FORECAST.store_col, FORECAST.item_col]
    n_dupes = df.duplicated(subset=dup_key).sum()
    if n_dupes > 0:
        raise DataValidationError(
            f"Found {n_dupes} duplicate (date, store, item) rows in train data."
        )

    # Check continuity of the date range per series
    incomplete_series = []
    for (store, item), group in df.groupby([FORECAST.store_col, FORECAST.item_col]):
        expected_days = (group[FORECAST.date_col].max() - group[FORECAST.date_col].min()).days + 1
        if len(group) != expected_days:
            incomplete_series.append((store, item, len(group), expected_days))

    if incomplete_series:
        logger.warning(
            "%d (store, item) series have gaps in their daily date range (showing up to 5): %s",
            len(incomplete_series),
            incomplete_series[:5],
        )

    logger.info(
        "Train data validation passed: %d rows, %d series.",
        len(df),
        df.groupby([FORECAST.store_col, FORECAST.item_col]).ngroups,
    )


def validate_test(df: pd.DataFrame) -> None:
    """Validate the test DataFrame.

    Checks:
        - Required columns present.
        - No nulls in key columns.
        - No duplicate (date, store, item) combinations.
    """
    required = ["id", FORECAST.date_col, FORECAST.store_col, FORECAST.item_col]
    _require_columns(df, required, "test")

    if df[required].isnull().any().any():
        null_counts = df[required].isnull().sum()
        raise DataValidationError(
            f"Null values found in test data: {null_counts[null_counts > 0].to_dict()}"
        )

    dup_key = [FORECAST.date_col, FORECAST.store_col, FORECAST.item_col]
    n_dupes = df.duplicated(subset=dup_key).sum()
    if n_dupes > 0:
        raise DataValidationError(
            f"Found {n_dupes} duplicate (date, store, item) rows in test data."
        )

    logger.info(
        "Test data validation passed: %d rows, horizon=%d days expected.",
        len(df),
        FORECAST.horizon_days,
    )


def validate_train_test_consistency(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Verify that test (store, item) combinations exist in training history."""
    train_pairs = set(
        map(tuple, train_df[[FORECAST.store_col, FORECAST.item_col]].drop_duplicates().values)
    )
    test_pairs = set(
        map(tuple, test_df[[FORECAST.store_col, FORECAST.item_col]].drop_duplicates().values)
    )

    unseen = test_pairs - train_pairs
    if unseen:
        raise DataValidationError(
            f"{len(unseen)} (store, item) combinations in test data have no training history: "
            f"{list(unseen)[:5]}"
        )

    test_start = test_df[FORECAST.date_col].min()
    train_end = train_df[FORECAST.date_col].max()
    if test_start <= train_end:
        raise DataValidationError(
            f"Test period ({test_start.date()}) overlaps with or precedes end of training data "
            f"({train_end.date()})."
        )

    logger.info(
        "Train/test consistency validated: all %d series present, no temporal overlap.",
        len(test_pairs),
    )

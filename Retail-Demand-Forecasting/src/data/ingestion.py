"""Data ingestion: load raw train/test CSVs into typed pandas DataFrames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.settings import FORECAST, PATHS
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_TRAIN_DTYPES = {
    "store": "int16",
    "item": "int16",
    "sales": "int32",
}
_TEST_DTYPES = {
    "id": "int32",
    "store": "int16",
    "item": "int16",
}


def load_train(path: Path | None = None) -> pd.DataFrame:
    """Load the raw training data (date, store, item, sales).

    Args:
        path: Optional override for the CSV location; defaults to
            ``PATHS.train_csv``.

    Returns:
        DataFrame with parsed dates and downcast dtypes, sorted by
        (store, item, date).
    """
    csv_path = path or PATHS.train_csv
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {csv_path}. Place train.csv under data/raw/."
        )

    logger.info("Loading training data from %s", csv_path)
    df = pd.read_csv(csv_path, dtype=_TRAIN_DTYPES, parse_dates=[FORECAST.date_col])
    df = df.sort_values([FORECAST.store_col, FORECAST.item_col, FORECAST.date_col]).reset_index(
        drop=True
    )
    logger.info(
        "Loaded %d training rows spanning %s to %s",
        len(df),
        df[FORECAST.date_col].min(),
        df[FORECAST.date_col].max(),
    )
    return df


def load_test(path: Path | None = None) -> pd.DataFrame:
    """Load the raw test/scoring data (id, date, store, item).

    Args:
        path: Optional override for the CSV location; defaults to
            ``PATHS.test_csv``.

    Returns:
        DataFrame with parsed dates and downcast dtypes, sorted by
        (store, item, date).
    """
    csv_path = path or PATHS.test_csv
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {csv_path}. Place test.csv under data/raw/."
        )

    logger.info("Loading test data from %s", csv_path)
    df = pd.read_csv(csv_path, dtype=_TEST_DTYPES, parse_dates=[FORECAST.date_col])
    df = df.sort_values([FORECAST.store_col, FORECAST.item_col, FORECAST.date_col]).reset_index(
        drop=True
    )
    logger.info(
        "Loaded %d test rows spanning %s to %s",
        len(df),
        df[FORECAST.date_col].min(),
        df[FORECAST.date_col].max(),
    )
    return df


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience loader returning ``(train_df, test_df)``."""
    return load_train(), load_test()

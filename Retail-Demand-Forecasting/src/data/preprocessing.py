"""Preprocessing: fill calendar gaps and produce the time-based train/valid split."""

from __future__ import annotations

import pandas as pd

from src.config.settings import FORECAST
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def fill_calendar_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex every (store, item) series onto a complete daily calendar.

    Any missing days are filled with ``sales = 0`` (no recorded sale that
    day), which is the correct assumption for POS-style daily sales data.

    Args:
        df: Training DataFrame with columns [date, store, item, sales].

    Returns:
        DataFrame with one row per (store, item, day) across the full
        observed date range, sorted by (store, item, date).
    """
    date_col, store_col, item_col, target_col = (
        FORECAST.date_col,
        FORECAST.store_col,
        FORECAST.item_col,
        FORECAST.target_col,
    )

    full_range = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    filled_groups = []

    for (store, item), group in df.groupby([store_col, item_col]):
        group = group.set_index(date_col).reindex(full_range)
        group[store_col] = store
        group[item_col] = item
        group[target_col] = group[target_col].fillna(0)
        group.index.name = date_col
        filled_groups.append(group.reset_index())

    result = pd.concat(filled_groups, ignore_index=True)
    result = result.sort_values([store_col, item_col, date_col]).reset_index(drop=True)

    added_rows = len(result) - len(df)
    if added_rows > 0:
        logger.info("Filled %d missing calendar-day rows across all series.", added_rows)
    else:
        logger.info("No calendar gaps found; data already fully continuous.")

    return result


def time_based_train_valid_split(
    df: pd.DataFrame, validation_days: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a feature-complete DataFrame into train/validation by date.

    The most recent ``validation_days`` days of history are held out as
    the validation set, which mimics the real forecasting task (predicting
    a future window) far better than a random row split would.

    Args:
        df: DataFrame with a date column, already feature-engineered.
        validation_days: Size of the held-out window in days; defaults to
            ``ForecastConfig.validation_days``.

    Returns:
        ``(train_df, valid_df)`` tuple.
    """
    validation_days = validation_days or FORECAST.validation_days
    cutoff = df[FORECAST.date_col].max() - pd.Timedelta(days=validation_days - 1)

    train_df = df[df[FORECAST.date_col] < cutoff].reset_index(drop=True)
    valid_df = df[df[FORECAST.date_col] >= cutoff].reset_index(drop=True)

    logger.info(
        "Time-based split: train=%d rows (< %s), valid=%d rows (>= %s)",
        len(train_df),
        cutoff.date(),
        len(valid_df),
        cutoff.date(),
    )
    return train_df, valid_df

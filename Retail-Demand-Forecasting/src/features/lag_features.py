"""Lag / rolling-window features for a fixed 90-day-ahead forecast.

CRITICAL DESIGN NOTE ON LEAKAGE
--------------------------------
This project forecasts a full 90-day horizon in one shot ("direct"
multi-step forecasting), not one day at a time. That means that when the
model predicts sales for the *last* day of the horizon, none of the last
89 days of "recent" history are actually available yet at prediction
time — only data up to and including ``target_date - horizon_days`` is
guaranteed to exist for every point in the horizon.

To keep every lag/rolling feature usable across the *entire* 90-day
horizon without recursion, every feature here is computed relative to an
"as-of" date defined as::

    as_of_date = target_date - horizon_days

A "lag_k" feature is therefore the sales value ``k`` days before
``as_of_date`` (i.e. ``target_date - horizon_days - k``), and a
"rolling_w" feature is a statistic over the ``w``-day window ending at
``as_of_date``. This guarantees the exact same feature computation logic
applies identically whether we are training, validating, or scoring the
true 2018 test set.
"""

from __future__ import annotations

import pandas as pd

from src.config.settings import FORECAST
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    lag_days: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    horizon_days: int | None = None,
) -> pd.DataFrame:
    """Add horizon-safe lag and rolling-window features per (store, item).

    Args:
        df: Full history DataFrame with columns [date, store, item, sales],
            one row per (store, item, day), sorted or unsorted.
        lag_days: Extra offsets (in days) behind the as-of date to lag by.
            Defaults to ``ForecastConfig.lag_days``.
        rolling_windows: Window sizes (in days) for rolling mean/std/min/max,
            computed ending at the as-of date. Defaults to
            ``ForecastConfig.rolling_windows``.
        horizon_days: Forecast horizon used to compute the as-of date.
            Defaults to ``ForecastConfig.horizon_days``.

    Returns:
        DataFrame sorted by (store, item, date) with lag_*, roll_*_mean,
        roll_*_std, roll_*_min, roll_*_max, and roll_*_median columns added.
        Rows without enough history for a given feature contain NaN for
        that feature (the tree models used downstream handle NaN natively).
    """
    lag_days = lag_days or FORECAST.lag_days
    rolling_windows = rolling_windows or FORECAST.rolling_windows
    horizon_days = horizon_days if horizon_days is not None else FORECAST.horizon_days

    date_col, store_col, item_col, target_col = (
        FORECAST.date_col,
        FORECAST.store_col,
        FORECAST.item_col,
        FORECAST.target_col,
    )

    out = df.sort_values([store_col, item_col, date_col]).reset_index(drop=True).copy()
    grouped_sales = out.groupby([store_col, item_col])[target_col]

    # The as-of date for a row at position i is (target_date - horizon_days).
    # In row-shift terms (data is daily and fully continuous), that is a
    # shift of `horizon_days` rows within each (store, item) group. Every
    # additional lag/rolling computation happens *on top of* that base shift.
    base_shifted = grouped_sales.shift(horizon_days)

    for lag in lag_days:
        # lag_k means "k days before the as-of date" -> total shift = horizon + lag
        out[f"lag_{lag}"] = grouped_sales.shift(horizon_days + lag)

    # Rolling stats computed on the horizon-shifted series so the window
    # ends exactly at the as-of date.
    grouped_shifted = base_shifted.groupby([out[store_col], out[item_col]])
    for window in rolling_windows:
        roll = grouped_shifted.rolling(window=window, min_periods=max(3, window // 3))
        out[f"roll_{window}_mean"] = roll.mean().reset_index(level=[0, 1], drop=True)
        out[f"roll_{window}_std"] = roll.std().reset_index(level=[0, 1], drop=True)
        out[f"roll_{window}_min"] = roll.min().reset_index(level=[0, 1], drop=True)
        out[f"roll_{window}_max"] = roll.max().reset_index(level=[0, 1], drop=True)
        out[f"roll_{window}_median"] = roll.median().reset_index(level=[0, 1], drop=True)

    # Trend signal: ratio of a short vs long rolling mean at the as-of date,
    # useful for tree models to detect recent acceleration/deceleration.
    if 7 in rolling_windows and 28 in rolling_windows:
        out["roll_trend_7_28"] = (
            out["roll_7_mean"] / out["roll_28_mean"].replace(0, pd.NA)
        ).astype("float32")

    n_feature_cols = (
        len(lag_days) + 5 * len(rolling_windows) + (1 if "roll_trend_7_28" in out.columns else 0)
    )
    logger.info(
        "Added %d lag/rolling features (horizon-safe, as-of = target_date - %d days).",
        n_feature_cols,
        horizon_days,
    )

    return out

"""Calendar-derived features: day-of-week, seasonality, holidays-style flags."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import FORECAST


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-derived columns computed purely from the date column.

    These features are available at prediction time for any future date,
    so they carry no leakage risk (unlike lag/rolling features, which
    must be computed carefully — see ``lag_features.py``).

    Added columns:
        year, month, day, day_of_week, day_of_year, week_of_year,
        is_weekend, is_month_start, is_month_end, quarter,
        month_sin, month_cos, dow_sin, dow_cos (cyclical encodings).

    Args:
        df: DataFrame containing the configured date column.

    Returns:
        A copy of ``df`` with calendar features appended.
    """
    out = df.copy()
    date_col = FORECAST.date_col
    dt = out[date_col]

    out["year"] = dt.dt.year.astype("int16")
    out["month"] = dt.dt.month.astype("int8")
    out["day"] = dt.dt.day.astype("int8")
    out["day_of_week"] = dt.dt.dayofweek.astype("int8")
    out["day_of_year"] = dt.dt.dayofyear.astype("int16")
    out["week_of_year"] = dt.dt.isocalendar().week.astype("int16")
    out["quarter"] = dt.dt.quarter.astype("int8")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype("int8")
    out["is_month_start"] = dt.dt.is_month_start.astype("int8")
    out["is_month_end"] = dt.dt.is_month_end.astype("int8")

    # Cyclical encodings so the model sees December/January and
    # Sunday/Monday as numerically adjacent.
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12).astype("float32")
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12).astype("float32")
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7).astype("float32")
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7).astype("float32")

    return out

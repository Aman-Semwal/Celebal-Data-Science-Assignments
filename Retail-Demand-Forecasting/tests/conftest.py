"""Shared pytest fixtures: a small synthetic dataset for fast, deterministic tests.

Unit tests should never depend on the real (913K-row) dataset -- that
would make the test suite slow and non-portable. This fixture builds a
tiny but structurally valid (date, store, item, sales) DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_train_df() -> pd.DataFrame:
    """A small, fully continuous (date, store, item, sales) DataFrame: 2 stores x 3 items x 400 days."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    rows = []
    for store in [1, 2]:
        for item in [1, 2, 3]:
            base = 10 + store * 2 + item
            seasonal = 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
            noise = rng.normal(0, 1, len(dates))
            sales = np.clip(base + seasonal + noise, 0, None).round().astype(int)
            for d, s in zip(dates, sales, strict=True):
                rows.append({"date": d, "store": store, "item": item, "sales": s})
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_test_df(synthetic_train_df: pd.DataFrame) -> pd.DataFrame:
    """A small test DataFrame covering the 90 days immediately after ``synthetic_train_df``."""
    last_date = synthetic_train_df["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=90, freq="D")
    rows = []
    idx = 0
    for store in [1, 2]:
        for item in [1, 2, 3]:
            for d in future_dates:
                rows.append({"id": idx, "date": d, "store": store, "item": item})
                idx += 1
    return pd.DataFrame(rows)

"""Common utility functions shared across the pipeline."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Seed Python's ``random`` and NumPy for reproducibility.

    Note: individual model libraries (LightGBM, XGBoost, CatBoost, torch)
    accept their own ``random_state``/``seed`` arguments, which are set
    from ``ForecastConfig.random_seed`` / ``ModelDefaults`` where relevant.
    """
    random.seed(seed)
    np.random.seed(seed)


@contextmanager
def timer(task_name: str) -> Iterator[None]:
    """Context manager that logs the wall-clock duration of a code block.

    Example:
        with timer("feature engineering"):
            build_features(df)
    """
    start = time.perf_counter()
    logger.info("Starting: %s", task_name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Finished: %s (%.2fs)", task_name, elapsed)


def save_json(obj: Any, path: Path) -> None:
    """Serialize ``obj`` to JSON at ``path``, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    """Load a JSON file from ``path``."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def reduce_memory_usage(df, verbose: bool = True):
    """Downcast numeric columns of a DataFrame to reduce memory footprint.

    Args:
        df: Input pandas DataFrame.
        verbose: Whether to log the memory reduction achieved.

    Returns:
        The same DataFrame with numeric columns downcast in place.
    """
    import pandas as pd

    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if (
            col_type == np.dtype("object")
            or isinstance(col_type, pd.CategoricalDtype)
            or pd.api.types.is_datetime64_any_dtype(col_type)
            or pd.api.types.is_bool_dtype(col_type)
        ):
            continue
        c_min, c_max = df[col].min(), df[col].max()
        if str(col_type)[:3] == "int":
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        else:
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        logger.info(
            "Memory usage reduced from %.2f MB to %.2f MB (-%.1f%%)",
            start_mem,
            end_mem,
            100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0.0,
        )
    return df

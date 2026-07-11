"""Forecast accuracy metrics used to compare every model on the same footing."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true, dtype="float64"), np.asarray(y_pred, dtype="float64")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true, y_pred = np.asarray(y_true, dtype="float64"), np.asarray(y_pred, dtype="float64")
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """Mean Absolute Percentage Error (%), with an epsilon to avoid divide-by-zero on days with 0 sales."""
    y_true, y_pred = np.asarray(y_true, dtype="float64"), np.asarray(y_pred, dtype="float64")
    denom = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """Symmetric Mean Absolute Percentage Error (%)."""
    y_true, y_pred = np.asarray(y_true, dtype="float64"), np.asarray(y_pred, dtype="float64")
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (%): total absolute error / total actual volume.

    More robust than MAPE/SMAPE for intermittent or low-volume series
    since it aggregates error across all series before dividing.
    """
    y_true, y_pred = np.asarray(y_true, dtype="float64"), np.asarray(y_pred, dtype="float64")
    total_actual = np.sum(np.abs(y_true))
    if total_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100)


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the full metric suite for a set of predictions.

    Returns:
        Dict with keys: rmse, mae, mape, smape, wape.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
    }

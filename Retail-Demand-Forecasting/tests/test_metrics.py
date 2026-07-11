"""Tests for src.evaluation.metrics."""

from __future__ import annotations

import numpy as np
from src.evaluation.metrics import evaluate_all, mae, mape, rmse, smape, wape


def test_rmse_zero_for_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_mae_zero_for_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    assert np.isclose(rmse(y_true, y_pred), np.sqrt((9 + 16) / 2))


def test_wape_known_value():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    # total abs error = 2 + 2 = 4, total actual = 30 -> wape = 4/30*100
    assert np.isclose(wape(y_true, y_pred), 4 / 30 * 100)


def test_wape_zero_when_actuals_are_zero():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 2.0])
    assert wape(y_true, y_pred) == 0.0


def test_mape_handles_zero_actuals_with_epsilon():
    y_true = np.array([0.0])
    y_pred = np.array([5.0])
    result = mape(y_true, y_pred, epsilon=1.0)
    assert np.isfinite(result)


def test_smape_symmetric():
    y_a = np.array([10.0])
    y_b = np.array([12.0])
    assert np.isclose(smape(y_a, y_b), smape(y_b, y_a))


def test_evaluate_all_returns_all_keys():
    y = np.array([1.0, 2.0, 3.0])
    result = evaluate_all(y, y)
    assert set(result.keys()) == {"rmse", "mae", "mape", "smape", "wape"}

"""Segment-level evaluation: breaks down accuracy by store, item, and day-of-horizon.

Aggregate metrics (WAPE, RMSE, ...) can hide systematic weaknesses in a
model — e.g. a model might do well on average but be consistently bad on
low-volume items, which matters a lot for inventory decisions on those
SKUs specifically. This module surfaces those breakdowns.
"""

from __future__ import annotations

import pandas as pd

from src.config.settings import FORECAST, PATHS
from src.evaluation.metrics import evaluate_all
from src.utils.common import save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_by_segment(
    df: pd.DataFrame, y_true_col: str, y_pred_col: str, segment_col: str
) -> pd.DataFrame:
    """Compute the full metric suite within each level of ``segment_col``.

    Args:
        df: DataFrame containing true values, predictions, and the segment column.
        y_true_col: Column name holding actual sales.
        y_pred_col: Column name holding predicted sales.
        segment_col: Column to group by (e.g. 'store', 'item').

    Returns:
        DataFrame indexed by segment value, one row of metrics per segment,
        sorted by WAPE descending (worst segments first).
    """
    rows = []
    for segment_value, group in df.groupby(segment_col, observed=True):
        metrics = evaluate_all(group[y_true_col].to_numpy(), group[y_pred_col].to_numpy())
        metrics[segment_col] = segment_value
        metrics["n_obs"] = len(group)
        rows.append(metrics)
    result = pd.DataFrame(rows).set_index(segment_col).sort_values("wape", ascending=False)
    return result


def evaluate_by_horizon_day(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> pd.DataFrame:
    """Compute metrics for each day-offset within the 90-day forecast horizon.

    This reveals whether accuracy degrades further out in the horizon
    (common in demand forecasting) — critical for deciding how far out
    inventory decisions can safely rely on the forecast.

    Args:
        df: DataFrame with a date column, true values, and predictions,
            covering exactly one contiguous horizon window.
        y_true_col: Column name holding actual sales.
        y_pred_col: Column name holding predictions.

    Returns:
        DataFrame indexed by ``horizon_day`` (1..N), one row of metrics per day-offset.
    """
    tmp = df.copy()
    min_date = tmp[FORECAST.date_col].min()
    tmp["horizon_day"] = (tmp[FORECAST.date_col] - min_date).dt.days + 1

    rows = []
    for day, group in tmp.groupby("horizon_day"):
        metrics = evaluate_all(group[y_true_col].to_numpy(), group[y_pred_col].to_numpy())
        metrics["horizon_day"] = day
        metrics["n_obs"] = len(group)
        rows.append(metrics)
    return pd.DataFrame(rows).set_index("horizon_day").sort_index()


def run_full_evaluation_report(model_name: str, model, feature_columns: list[str]) -> dict:
    """Generate and persist store-level, item-level, and horizon-day breakdowns for one model.

    Args:
        model_name: Identifier used in output filenames.
        model: A fitted model exposing ``.predict(X) -> np.ndarray``.
        feature_columns: Feature columns the model expects.

    Returns:
        Dict with summary statistics (worst/best store and item, horizon degradation).
    """
    valid = pd.read_parquet(PATHS.valid_features_parquet)
    preds = model.predict(valid[feature_columns])
    valid = valid.copy()
    valid["y_pred"] = preds

    by_store = evaluate_by_segment(valid, FORECAST.target_col, "y_pred", FORECAST.store_col)
    by_item = evaluate_by_segment(valid, FORECAST.target_col, "y_pred", FORECAST.item_col)
    by_horizon = evaluate_by_horizon_day(valid, FORECAST.target_col, "y_pred")

    by_store.to_csv(PATHS.reports_dir / f"{model_name}_eval_by_store.csv")
    by_item.to_csv(PATHS.reports_dir / f"{model_name}_eval_by_item.csv")
    by_horizon.to_csv(PATHS.reports_dir / f"{model_name}_eval_by_horizon_day.csv")

    summary = {
        "worst_store": {"store": str(by_store.index[0]), "wape": float(by_store.iloc[0]["wape"])},
        "best_store": {"store": str(by_store.index[-1]), "wape": float(by_store.iloc[-1]["wape"])},
        "worst_item": {"item": str(by_item.index[0]), "wape": float(by_item.iloc[0]["wape"])},
        "best_item": {"item": str(by_item.index[-1]), "wape": float(by_item.iloc[-1]["wape"])},
        "horizon_day_1_wape": float(by_horizon.iloc[0]["wape"]),
        "horizon_day_90_wape": float(by_horizon.iloc[-1]["wape"]),
    }
    save_json(summary, PATHS.reports_dir / f"{model_name}_eval_summary.json")
    logger.info("Segment evaluation for %s: %s", model_name, summary)
    return summary

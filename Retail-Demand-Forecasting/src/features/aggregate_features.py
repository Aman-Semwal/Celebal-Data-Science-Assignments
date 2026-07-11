"""Store/item aggregate features: series-level and cross-series statistics.

These capture "how big is this store" / "how popular is this item"
signals that plain lags cannot express well, especially for tree models
splitting across hundreds of series at once.
"""

from __future__ import annotations

import pandas as pd

from src.config.settings import FORECAST


def add_store_item_aggregates(
    df: pd.DataFrame, reference_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Add store-level, item-level, and store x item aggregate statistics.

    Aggregates are computed from ``reference_df`` (defaults to ``df``
    itself) so that, when scoring a held-out or test set, the caller can
    pass the training history as ``reference_df`` to avoid leaking future
    information into the aggregates.

    Added columns:
        store_mean_sales, store_std_sales,
        item_mean_sales, item_std_sales,
        store_item_mean_sales, store_item_std_sales,
        item_share_of_store (item's mean sales as a fraction of its store's
        total mean sales across items).

    Args:
        df: DataFrame to attach aggregate features to.
        reference_df: DataFrame to compute the aggregate statistics from.
            Defaults to ``df``.

    Returns:
        A copy of ``df`` with aggregate feature columns appended.
    """
    ref = reference_df if reference_df is not None else df
    store_col, item_col, target_col = FORECAST.store_col, FORECAST.item_col, FORECAST.target_col

    store_stats = ref.groupby(store_col)[target_col].agg(
        store_mean_sales="mean", store_std_sales="std"
    )
    item_stats = ref.groupby(item_col)[target_col].agg(item_mean_sales="mean", item_std_sales="std")
    store_item_stats = ref.groupby([store_col, item_col])[target_col].agg(
        store_item_mean_sales="mean", store_item_std_sales="std"
    )

    out = df.merge(store_stats, on=store_col, how="left")
    out = out.merge(item_stats, on=item_col, how="left")
    out = out.merge(store_item_stats, on=[store_col, item_col], how="left")

    store_totals = out.groupby(store_col)["store_item_mean_sales"].transform("sum")
    out["item_share_of_store"] = (out["store_item_mean_sales"] / store_totals).astype("float32")

    return out

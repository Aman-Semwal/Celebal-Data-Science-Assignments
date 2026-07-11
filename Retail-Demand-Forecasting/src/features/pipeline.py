"""End-to-end feature engineering pipeline.

Combines calendar features, horizon-safe lag/rolling features, and
store/item aggregates into a single model-ready feature table, and
persists the result plus the resolved feature column list.
"""

from __future__ import annotations

import pandas as pd

from src.config.settings import FORECAST, PATHS
from src.data.ingestion import load_test, load_train
from src.data.preprocessing import fill_calendar_gaps, time_based_train_valid_split
from src.data.validation import validate_test, validate_train, validate_train_test_consistency
from src.features.aggregate_features import add_store_item_aggregates
from src.features.calendar_features import add_calendar_features
from src.features.lag_features import add_lag_and_rolling_features
from src.utils.common import reduce_memory_usage, save_json, timer
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Non-feature identifier/target columns excluded from the model matrix.
_ID_COLUMNS = [FORECAST.date_col, FORECAST.store_col, FORECAST.item_col, FORECAST.target_col, "id"]


def build_feature_table(
    train_df: pd.DataFrame, reference_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Apply the full feature pipeline to a (store, item, date, sales) table.

    Args:
        train_df: DataFrame to featurize (already calendar-gap-filled).
        reference_df: DataFrame used to compute aggregate statistics from
            (should be training history only, to avoid leakage). Defaults
            to ``train_df`` itself.

    Returns:
        Feature-complete DataFrame, memory-optimized.
    """
    df = add_calendar_features(train_df)
    df = add_lag_and_rolling_features(df)
    df = add_store_item_aggregates(
        df, reference_df=reference_df if reference_df is not None else train_df
    )
    df[FORECAST.store_col] = df[FORECAST.store_col].astype("category")
    df[FORECAST.item_col] = df[FORECAST.item_col].astype("category")
    df = reduce_memory_usage(df, verbose=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model feature column names (everything except IDs/target)."""
    return [c for c in df.columns if c not in _ID_COLUMNS]


def run_feature_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Run ingestion -> validation -> preprocessing -> features end to end.

    Returns:
        ``(train_features, valid_features, test_features, feature_columns)``
        where train/valid come from a time-based split of history, and
        test_features covers the true 2018 90-day scoring window.
    """
    with timer("data ingestion"):
        train_raw = load_train()
        test_raw = load_test()

    with timer("data validation"):
        validate_train(train_raw)
        validate_test(test_raw)
        validate_train_test_consistency(train_raw, test_raw)

    with timer("calendar gap filling"):
        train_filled = fill_calendar_gaps(train_raw)

    # Build test rows as an extension of the continuous daily history so
    # that lag/rolling features can be computed with the same function.
    with timer("assembling full history (train + test horizon)"):
        test_scaffold = test_raw[[FORECAST.date_col, FORECAST.store_col, FORECAST.item_col]].copy()
        test_scaffold[FORECAST.target_col] = pd.NA
        full_history = (
            pd.concat(
                [
                    train_filled[
                        [
                            FORECAST.date_col,
                            FORECAST.store_col,
                            FORECAST.item_col,
                            FORECAST.target_col,
                        ]
                    ],
                    test_scaffold,
                ],
                ignore_index=True,
            )
            .sort_values([FORECAST.store_col, FORECAST.item_col, FORECAST.date_col])
            .reset_index(drop=True)
        )

    with timer("feature engineering"):
        featured = build_feature_table(full_history, reference_df=train_filled)

    test_ids = test_raw[[FORECAST.date_col, FORECAST.store_col, FORECAST.item_col, "id"]]
    test_features = featured.merge(
        test_ids, on=[FORECAST.date_col, FORECAST.store_col, FORECAST.item_col], how="inner"
    )
    train_valid_features = featured[featured[FORECAST.target_col].notna()].reset_index(drop=True)

    with timer("time-based train/valid split"):
        train_features, valid_features = time_based_train_valid_split(train_valid_features)

    feature_columns = get_feature_columns(train_features)

    PATHS.ensure_dirs()
    train_features.to_parquet(PATHS.train_features_parquet, index=False)
    valid_features.to_parquet(PATHS.valid_features_parquet, index=False)
    test_features.to_parquet(PATHS.test_features_parquet, index=False)
    save_json(feature_columns, PATHS.data_processed / "feature_columns.json")

    logger.info(
        "Feature pipeline complete: train=%s valid=%s test=%s, %d feature columns.",
        train_features.shape,
        valid_features.shape,
        test_features.shape,
        len(feature_columns),
    )
    return train_features, valid_features, test_features, feature_columns


if __name__ == "__main__":
    run_feature_pipeline()

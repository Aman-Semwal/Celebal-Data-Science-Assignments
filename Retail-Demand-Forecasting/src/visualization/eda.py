"""Exploratory data analysis: summary statistics and static plots.

Running this module populates ``reports/eda_summary.json`` and PNG plots
under ``images/`` directly from the raw training data.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config.settings import FORECAST, PATHS
from src.data.ingestion import load_train
from src.utils.common import save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def compute_summary_stats(df: pd.DataFrame) -> dict:
    """Compute headline dataset statistics used in the README/report."""
    date_col, store_col, item_col, target_col = (
        FORECAST.date_col,
        FORECAST.store_col,
        FORECAST.item_col,
        FORECAST.target_col,
    )
    return {
        "n_rows": int(len(df)),
        "n_stores": int(df[store_col].nunique()),
        "n_items": int(df[item_col].nunique()),
        "n_series": int(df.groupby([store_col, item_col]).ngroups),
        "date_min": str(df[date_col].min().date()),
        "date_max": str(df[date_col].max().date()),
        "sales_mean": float(df[target_col].mean()),
        "sales_std": float(df[target_col].std()),
        "sales_min": int(df[target_col].min()),
        "sales_max": int(df[target_col].max()),
        "sales_median": float(df[target_col].median()),
    }


def plot_overall_trend(df: pd.DataFrame, out_path) -> None:
    """Plot total daily sales across all stores/items over time."""
    daily = df.groupby(FORECAST.date_col)[FORECAST.target_col].sum()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily.index, daily.values, color="#2C6E9E", linewidth=0.8)
    ax.set_title("Total Daily Sales Across All Stores & Items (2013-2017)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Units Sold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_monthly_seasonality(df: pd.DataFrame, out_path) -> None:
    """Boxplot of sales distribution by calendar month."""
    tmp = df.copy()
    tmp["month"] = tmp[FORECAST.date_col].dt.month
    fig, ax = plt.subplots(figsize=(9, 4))
    tmp.boxplot(column=FORECAST.target_col, by="month", ax=ax, showfliers=False)
    ax.set_title("Sales Distribution by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Units Sold")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_dow_seasonality(df: pd.DataFrame, out_path) -> None:
    """Bar chart of mean sales by day of week."""
    tmp = df.copy()
    tmp["dow"] = tmp[FORECAST.date_col].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    means = tmp.groupby("dow")[FORECAST.target_col].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(means.index, means.values, color="#4C8C6B")
    ax.set_title("Mean Sales by Day of Week")
    ax.set_ylabel("Mean Units Sold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_store_item_heatmap(df: pd.DataFrame, out_path) -> None:
    """Heatmap of mean sales per (store, item), summarized by store x item-decile."""
    pivot = (
        df.groupby([FORECAST.store_col, FORECAST.item_col])[FORECAST.target_col].mean().unstack()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_title("Mean Daily Sales by Store (rows) x Item (columns)")
    ax.set_xlabel("Item")
    ax.set_ylabel("Store")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, label="Mean Units Sold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_top_bottom_items(df: pd.DataFrame, out_path) -> None:
    """Bar chart comparing the 5 best- and 5 worst-selling items on average."""
    item_means = df.groupby(FORECAST.item_col)[FORECAST.target_col].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4))
    combined = pd.concat([item_means.head(5), item_means.tail(5)])
    colors = ["#B0413E"] * 5 + ["#4C8C6B"] * 5
    ax.bar(combined.index.astype(str), combined.values, color=colors)
    ax.set_title("5 Lowest- vs 5 Highest-Selling Items (mean daily units)")
    ax.set_xlabel("Item ID")
    ax.set_ylabel("Mean Units Sold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_eda() -> dict:
    """Run the full EDA suite and persist stats + plots to disk."""
    PATHS.ensure_dirs()
    df = load_train()

    stats = compute_summary_stats(df)
    save_json(stats, PATHS.reports_dir / "eda_summary.json")
    logger.info("EDA summary stats: %s", stats)

    plot_overall_trend(df, PATHS.images_dir / "eda_overall_trend.png")
    plot_monthly_seasonality(df, PATHS.images_dir / "eda_monthly_seasonality.png")
    plot_dow_seasonality(df, PATHS.images_dir / "eda_dow_seasonality.png")
    plot_store_item_heatmap(df, PATHS.images_dir / "eda_store_item_heatmap.png")
    plot_top_bottom_items(df, PATHS.images_dir / "eda_top_bottom_items.png")

    logger.info("EDA plots saved to %s", PATHS.images_dir)
    return stats


if __name__ == "__main__":
    run_eda()

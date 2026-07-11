"""Time series analysis: STL decomposition and stationarity testing.

Applied to the aggregate (all-stores, all-items) daily series as a
representative example, and to one illustrative single (store, item)
series so the notebook/dashboard can show both scales.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller

from src.config.settings import FORECAST, PATHS
from src.data.ingestion import load_train
from src.utils.common import save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def run_stl_decomposition(series: pd.Series, period: int = 365) -> STL:
    """Run STL (Seasonal-Trend decomposition using LOESS) on a daily series.

    Args:
        series: Daily-frequency series indexed by date, no missing values.
        period: Seasonal period in days (365 for yearly seasonality).

    Returns:
        The fitted ``STLResult`` object (has ``.trend``, ``.seasonal``,
        ``.resid`` attributes).
    """
    stl = STL(series, period=period, robust=True)
    return stl.fit()


def run_adf_test(series: pd.Series) -> dict:
    """Run the Augmented Dickey-Fuller stationarity test.

    Returns:
        Dict with the test statistic, p-value, and a plain-language verdict.
    """
    result = adfuller(series.dropna())
    stat, p_value, *_ = result
    return {
        "adf_statistic": float(stat),
        "p_value": float(p_value),
        "is_stationary_at_5pct": bool(p_value < 0.05),
    }


def plot_stl_decomposition(stl_result, out_path, title: str) -> None:
    """Plot the observed/trend/seasonal/residual components from an STL fit."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(stl_result.observed, color="#2C6E9E", linewidth=0.8)
    axes[0].set_ylabel("Observed")
    axes[1].plot(stl_result.trend, color="#B0413E", linewidth=1.0)
    axes[1].set_ylabel("Trend")
    axes[2].plot(stl_result.seasonal, color="#4C8C6B", linewidth=0.6)
    axes[2].set_ylabel("Seasonal")
    axes[3].plot(stl_result.resid, color="#7A7A7A", linewidth=0.5)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Date")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_time_series_analysis() -> dict:
    """Run STL decomposition + ADF tests on the aggregate and one example series."""
    PATHS.ensure_dirs()
    df = load_train()

    aggregate = df.groupby(FORECAST.date_col)[FORECAST.target_col].sum()
    aggregate.index = pd.DatetimeIndex(aggregate.index, freq="D")

    example_mask = (df[FORECAST.store_col] == df[FORECAST.store_col].iloc[0]) & (
        df[FORECAST.item_col] == df[FORECAST.item_col].iloc[0]
    )
    example = df.loc[example_mask].set_index(FORECAST.date_col)[FORECAST.target_col]
    example.index = pd.DatetimeIndex(example.index, freq="D")

    agg_stl = run_stl_decomposition(aggregate)
    example_stl = run_stl_decomposition(example)

    plot_stl_decomposition(
        agg_stl, PATHS.images_dir / "stl_aggregate.png", "STL Decomposition: Aggregate Daily Sales"
    )
    plot_stl_decomposition(
        example_stl,
        PATHS.images_dir / "stl_example_series.png",
        f"STL Decomposition: Store {df[FORECAST.store_col].iloc[0]}, Item {df[FORECAST.item_col].iloc[0]}",
    )

    results = {
        "aggregate_adf": run_adf_test(aggregate),
        "example_series_adf": run_adf_test(example),
    }
    save_json(results, PATHS.reports_dir / "time_series_analysis.json")
    logger.info("Time series analysis results: %s", results)
    return results


if __name__ == "__main__":
    run_time_series_analysis()

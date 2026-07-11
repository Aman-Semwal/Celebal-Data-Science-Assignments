"""Model comparison plots built from ``reports/model_comparison.json``."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config.settings import PATHS
from src.utils.common import load_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

_BASELINE_NAMES = {"naive_last_value", "seasonal_naive", "moving_average"}


def plot_model_comparison(out_path=None) -> pd.DataFrame:
    """Bar-chart every model's WAPE/RMSE/MAE side by side, sorted by WAPE.

    Returns:
        The comparison table as a DataFrame (also useful for the dashboard).
    """
    report = load_json(PATHS.reports_dir / "model_comparison.json")
    df = pd.DataFrame(report).T.sort_values("wape")
    df["is_baseline"] = df.index.isin(_BASELINE_NAMES)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = df["is_baseline"].map({True: "#B0413E", False: "#2C6E9E"})

    for ax, metric, title in zip(
        axes, ["wape", "rmse", "mae"], ["WAPE (%)", "RMSE", "MAE"], strict=True
    ):
        ax.barh(df.index, df[metric], color=colors)
        ax.set_title(title)
        ax.invert_yaxis()

    fig.suptitle("Model Comparison on Held-Out Validation (90-day window) — red = baseline")
    fig.tight_layout()

    out_path = out_path or (PATHS.images_dir / "model_comparison.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Model comparison chart saved to %s", out_path)
    return df.drop(columns="is_baseline")


if __name__ == "__main__":
    print(plot_model_comparison())

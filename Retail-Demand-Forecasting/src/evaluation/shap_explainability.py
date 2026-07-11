"""SHAP explainability for the tree-based forecasting models.

Uses SHAP's TreeExplainer (exact, fast path for tree ensembles) to
compute feature attributions on a sample of the validation set, then
produces a global summary plot and persists raw SHAP values for the
dashboard's SHAP Analysis page.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config.settings import PATHS
from src.utils.common import load_json, save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def compute_shap_values(model, X_sample: pd.DataFrame):
    """Compute SHAP values for a tree-based model on a sample of rows.

    Args:
        model: A fitted model wrapper exposing ``.model`` (the underlying
            LightGBM/XGBoost/CatBoost/RandomForest estimator).
        X_sample: Feature DataFrame to explain (keep this small — a few
            hundred to a couple thousand rows — since SHAP scales with
            sample size).

    Returns:
        ``shap.Explanation`` object with values, base values, and data.
    """
    explainer = shap.TreeExplainer(model.model)
    explanation = explainer(X_sample)
    return explanation


def plot_shap_summary(explanation, out_path, title: str) -> None:
    """Save a SHAP beeswarm summary plot to ``out_path``."""
    fig = plt.figure(figsize=(9, 7))
    shap.plots.beeswarm(explanation, show=False, max_display=15)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_shap_importance_bar(explanation, out_path, title: str) -> None:
    """Save a SHAP mean-|value| bar chart (global feature importance) to ``out_path``."""
    fig = plt.figure(figsize=(9, 7))
    shap.plots.bar(explanation, show=False, max_display=15)
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def run_shap_analysis(
    model_name: str, model, X_valid: pd.DataFrame, sample_size: int = 1000
) -> dict:
    """Run the full SHAP analysis for one model and persist plots + a JSON summary.

    Args:
        model_name: Identifier used in output filenames.
        model: Fitted model wrapper (must expose ``.model`` — the raw estimator).
        X_valid: Validation feature matrix to sample rows from.
        sample_size: Number of rows to explain (SHAP cost scales with this).

    Returns:
        Dict mapping each feature to its mean absolute SHAP value, sorted descending.
    """
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_valid), size=min(sample_size, len(X_valid)), replace=False)
    X_sample = X_valid.iloc[sample_idx]

    explanation = compute_shap_values(model, X_sample)

    plot_shap_summary(
        explanation,
        PATHS.images_dir / f"shap_summary_{model_name}.png",
        f"SHAP Summary — {model_name}",
    )
    plot_shap_importance_bar(
        explanation,
        PATHS.images_dir / f"shap_importance_{model_name}.png",
        f"SHAP Feature Importance — {model_name}",
    )

    mean_abs_shap = pd.Series(
        np.abs(explanation.values).mean(axis=0), index=X_sample.columns
    ).sort_values(ascending=False)

    result = {
        "model_name": model_name,
        "sample_size": len(X_sample),
        "mean_abs_shap": mean_abs_shap.to_dict(),
    }
    save_json(result, PATHS.reports_dir / f"shap_{model_name}.json")
    logger.info(
        "SHAP analysis complete for %s. Top feature: %s", model_name, mean_abs_shap.index[0]
    )
    return result


if __name__ == "__main__":
    import argparse

    import pandas as pd

    from src.deployment.inference import _wrapper_for

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="catboost")
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()

    cols = load_json(PATHS.data_processed / "feature_columns.json")
    valid_df = pd.read_parquet(PATHS.valid_features_parquet)

    import joblib

    payload = joblib.load(PATHS.models_dir / f"{args.model}.joblib")
    wrapper = _wrapper_for(payload["name"])
    wrapper.model = payload["model"]
    wrapper.feature_columns = payload["feature_columns"]

    run_shap_analysis(args.model, wrapper, valid_df[cols], sample_size=args.sample_size)

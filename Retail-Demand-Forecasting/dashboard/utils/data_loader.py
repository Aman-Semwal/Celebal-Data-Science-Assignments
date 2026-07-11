"""Shared helpers used across every Streamlit dashboard page.

Centralizing cached loaders here means each page loads data/models once
per session instead of re-reading parquet/joblib files on every rerun.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the project's `src` package importable when Streamlit runs this
# file directly (Streamlit does not always respect PYTHONPATH the way a
# normal script invocation would).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.settings import PATHS  # noqa: E402
from src.utils.common import load_json  # noqa: E402


@st.cache_data(show_spinner="Loading training data...")
def load_raw_train() -> pd.DataFrame:
    from src.data.ingestion import load_train

    return load_train()


@st.cache_data(show_spinner="Loading feature tables...")
def load_feature_tables() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_parquet(PATHS.train_features_parquet)
    valid = pd.read_parquet(PATHS.valid_features_parquet)
    cols = load_json(PATHS.data_processed / "feature_columns.json")
    return train, valid, cols


@st.cache_data(show_spinner=False)
def load_model_comparison() -> dict:
    path = PATHS.reports_dir / "model_comparison.json"
    return load_json(path) if path.exists() else {}


@st.cache_resource(show_spinner="Loading trained model...")
def load_inference_service(model_name: str | None = None):
    from src.deployment.inference import InferenceService

    return InferenceService(model_name=model_name)


def available_saved_models() -> list[str]:
    """List which trained tree-based models exist on disk and are servable via InferenceService.

    Excludes ``attention_lstm`` (saved via a different format/wrapper not
    wired into the shared ``InferenceService``) and any ``*_mlflow``
    duplicate artifacts logged during MLflow runs.
    """
    if not PATHS.models_dir.exists():
        return []
    excluded = {"attention_lstm"}
    return sorted(
        p.stem
        for p in PATHS.models_dir.glob("*.joblib")
        if not p.stem.endswith("_mlflow") and p.stem not in excluded
    )

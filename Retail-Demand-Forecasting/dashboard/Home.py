"""Home page: project overview, key stats, and navigation guide."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.data_loader import (  # noqa: E402
    available_saved_models,
    load_model_comparison,
    load_raw_train,
)

st.set_page_config(page_title="Retail Demand Forecasting", page_icon="📦", layout="wide")

st.title("📦 Multi-Series Retail Demand Forecasting")
st.caption("Store x Item, 90-day-ahead demand forecasting for inventory optimization")

st.markdown("""
This dashboard explores an end-to-end forecasting system built on the
Kaggle **Store Item Demand Forecasting Challenge** dataset: 5 years of
daily sales across **10 stores × 50 items**, forecasting the next
**90 days** of demand for every series.
""")

try:
    df = load_raw_train()
    n_stores = df["store"].nunique()
    n_items = df["item"].nunique()
    n_series = df.groupby(["store", "item"]).ngroups

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stores", n_stores)
    col2.metric("Items", n_items)
    col3.metric("Series (Store x Item)", n_series)
    col4.metric("Daily Rows", f"{len(df):,}")

    st.divider()

    comparison = load_model_comparison()
    if comparison:
        best_model = min(comparison.items(), key=lambda kv: kv[1].get("wape", float("inf")))
        st.success(
            f"**Best model so far:** `{best_model[0]}` — WAPE **{best_model[1]['wape']:.2f}%**, "
            f"RMSE **{best_model[1]['rmse']:.2f}**"
        )

    models_available = available_saved_models()
    if models_available:
        st.info(f"**Trained models on disk:** {', '.join(models_available)}")
    else:
        st.warning(
            "No trained models found yet. Run `python -m src.models.train --model all` first."
        )

except FileNotFoundError:
    st.error(
        "Raw training data not found under `data/raw/train.csv`. Place the dataset there first."
    )

st.divider()
st.markdown("""
### Navigate using the sidebar

- **EDA** — dataset-wide exploratory analysis and seasonality
- **Forecast Explorer** — pick any store/item and see the 90-day forecast
- **Store Selection** / **Item Selection** — drill into per-segment accuracy
- **Model Comparison** — compare all trained models side by side
- **SHAP Analysis** — understand what drives each prediction
- **Download Forecast** — export forecasts as CSV
""")

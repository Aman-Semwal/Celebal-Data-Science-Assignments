"""Model Comparison page: compare every trained model side by side."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.settings import PATHS  # noqa: E402

from utils.data_loader import load_model_comparison  # noqa: E402

st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")
st.title("⚖️ Model Comparison")

report = load_model_comparison()
if not report:
    st.warning("No comparison report found. Run `python -m src.models.train --model all` first.")
    st.stop()

comp_df = pd.DataFrame(report).T
comp_df.index.name = "model"
comp_df = comp_df.sort_values("wape")

st.subheader("Validation Metrics")
st.dataframe(
    comp_df.style.highlight_min(subset=["rmse", "mae", "mape", "smape", "wape"], color="#c6f6d5"),
    width="stretch",
)

st.subheader("WAPE Comparison (lower is better)")
fig = px.bar(
    comp_df.reset_index(),
    x="model",
    y="wape",
    color="wape",
    color_continuous_scale="RdYlGn_r",
    title="Weighted Absolute Percentage Error by Model",
)
st.plotly_chart(fig, width="stretch")

st.subheader("Training Time vs. Accuracy Trade-off")
fig2 = px.scatter(
    comp_df.reset_index(),
    x="fit_seconds",
    y="wape",
    text="model",
    title="Fit Time vs. WAPE (bottom-left is best)",
)
fig2.update_traces(textposition="top center")
st.plotly_chart(fig2, width="stretch")

precomputed_chart = PATHS.images_dir / "model_comparison.png"
if precomputed_chart.exists():
    st.divider()
    st.subheader("Pre-generated Comparison Chart")
    st.image(str(precomputed_chart), width="stretch")

"""Forecast Explorer page: pick any (store, item) and view its 90-day forecast."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_loader import (  # noqa: E402
    available_saved_models,
    load_inference_service,
    load_raw_train,
)

st.set_page_config(page_title="Forecast Explorer", page_icon="📈", layout="wide")
st.title("📈 Forecast Explorer")

df = load_raw_train()
stores = sorted(df["store"].unique().tolist())
items = sorted(df["item"].unique().tolist())
models = available_saved_models()

col1, col2, col3 = st.columns(3)
store = col1.selectbox("Store", stores)
item = col2.selectbox("Item", items)
model_choice = col3.selectbox("Model", models if models else ["(none trained)"])

if not models:
    st.error(
        "No trained models found. Train at least one with `python -m src.models.train --model catboost`."
    )
    st.stop()

with st.spinner("Loading model and generating forecast..."):
    service = load_inference_service(model_choice)
    forecast_df = service.predict_horizon(store, item, horizon_days=90)

history = df[(df["store"] == store) & (df["item"] == item)].tail(180)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=history["date"], y=history["sales"], name="Historical Sales", line=dict(color="#2C6E9E")
    )
)
fig.add_trace(
    go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["predicted_sales"],
        name=f"{model_choice} Forecast (next 90 days)",
        line=dict(color="#B0413E", dash="dash"),
    )
)
fig.update_layout(
    title=f"Store {store}, Item {item} — Last 180 Days + 90-Day Forecast",
    xaxis_title="Date",
    yaxis_title="Units Sold",
    height=500,
)
st.plotly_chart(fig, width="stretch")

st.subheader("Forecast Table")
st.dataframe(forecast_df, width="stretch", hide_index=True)

st.session_state["last_forecast_df"] = forecast_df
st.session_state["last_forecast_meta"] = {"store": store, "item": item, "model": model_choice}

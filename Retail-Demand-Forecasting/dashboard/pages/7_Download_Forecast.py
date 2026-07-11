"""Download Forecast page: export a 90-day forecast (or a batch of them) as CSV."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_loader import (  # noqa: E402
    available_saved_models,
    load_inference_service,
    load_raw_train,
)

st.set_page_config(page_title="Download Forecast", page_icon="⬇️", layout="wide")
st.title("⬇️ Download Forecast")

df = load_raw_train()
stores = sorted(df["store"].unique().tolist())
items = sorted(df["item"].unique().tolist())
models = available_saved_models()

if not models:
    st.error("No trained models found. Train at least one first.")
    st.stop()

mode = st.radio(
    "What would you like to export?",
    ["Single (store, item) forecast", "Batch: all items for one store"],
)
model_choice = st.selectbox("Model", models)

if mode == "Single (store, item) forecast":
    col1, col2 = st.columns(2)
    store = col1.selectbox("Store", stores)
    item = col2.selectbox("Item", items)

    if st.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            service = load_inference_service(model_choice)
            forecast_df = service.predict_horizon(store, item, horizon_days=90)
            forecast_df.insert(0, "item", item)
            forecast_df.insert(0, "store", store)

        st.dataframe(forecast_df, width="stretch", hide_index=True)
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name=f"forecast_store{store}_item{item}.csv",
            mime="text/csv",
        )

else:
    store = st.selectbox("Store", stores)
    if st.button("Generate Forecasts for All Items in This Store"):
        with st.spinner(
            f"Generating forecasts for all {len(items)} items — this may take a moment..."
        ):
            service = load_inference_service(model_choice)
            all_forecasts = []
            progress = st.progress(0)
            for i, item in enumerate(items):
                fc = service.predict_horizon(store, item, horizon_days=90)
                fc.insert(0, "item", item)
                fc.insert(0, "store", store)
                all_forecasts.append(fc)
                progress.progress((i + 1) / len(items))
            batch_df = pd.concat(all_forecasts, ignore_index=True)

        st.success(f"Generated {len(batch_df)} forecast rows across {len(items)} items.")
        st.dataframe(batch_df.head(200), width="stretch", hide_index=True)
        csv = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Full Batch CSV",
            data=csv,
            file_name=f"forecast_store{store}_all_items.csv",
            mime="text/csv",
        )

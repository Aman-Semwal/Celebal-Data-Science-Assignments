"""Store Selection page: per-store validation accuracy and sales profile."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.segment_evaluation import evaluate_by_segment  # noqa: E402

from utils.data_loader import load_feature_tables, load_raw_train  # noqa: E402

st.set_page_config(page_title="Store Selection", page_icon="🏬", layout="wide")
st.title("🏬 Store-Level Analysis")

df = load_raw_train()
stores = sorted(df["store"].unique().tolist())
store = st.selectbox("Select a store", stores)

store_df = df[df["store"] == store]

col1, col2 = st.columns(2)
col1.metric("Mean Daily Sales (this store)", f"{store_df['sales'].mean():.2f}")
col2.metric("Total Units Sold (2013-2017)", f"{int(store_df['sales'].sum()):,}")

st.subheader("Sales by Item Within This Store")
item_means = store_df.groupby("item")["sales"].mean().sort_values(ascending=False).reset_index()
fig = px.bar(item_means, x="item", y="sales", title=f"Mean Daily Sales by Item — Store {store}")
st.plotly_chart(fig, width="stretch")

st.subheader("Daily Sales Trend for This Store")
daily = store_df.groupby("date")["sales"].sum().reset_index()
fig2 = px.line(daily, x="date", y="sales", title=f"Total Daily Sales — Store {store}")
st.plotly_chart(fig2, width="stretch")

st.divider()
st.subheader("Model Accuracy for This Store (Validation Set)")
try:
    _, valid, cols = load_feature_tables()
    from src.deployment.inference import get_inference_service

    service = get_inference_service()
    valid_store = valid[valid["store"] == store]
    if len(valid_store) > 0:
        preds = service.model.predict(valid_store[cols])
        valid_store = valid_store.copy()
        valid_store["y_pred"] = preds
        metrics_df = evaluate_by_segment(valid_store, "sales", "y_pred", "store")
        st.dataframe(metrics_df, width="stretch")
    else:
        st.info("No validation rows for this store in the current split.")
except FileNotFoundError:
    st.warning("No trained model found yet — accuracy breakdown unavailable.")

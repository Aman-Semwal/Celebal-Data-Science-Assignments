"""Item Selection page: per-item validation accuracy and sales profile across stores."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.segment_evaluation import evaluate_by_segment  # noqa: E402

from utils.data_loader import load_feature_tables, load_raw_train  # noqa: E402

st.set_page_config(page_title="Item Selection", page_icon="🛒", layout="wide")
st.title("🛒 Item-Level Analysis")

df = load_raw_train()
items = sorted(df["item"].unique().tolist())
item = st.selectbox("Select an item", items)

item_df = df[df["item"] == item]

col1, col2 = st.columns(2)
col1.metric("Mean Daily Sales (this item)", f"{item_df['sales'].mean():.2f}")
col2.metric("Total Units Sold (2013-2017)", f"{int(item_df['sales'].sum()):,}")

st.subheader("Sales by Store for This Item")
store_means = item_df.groupby("store")["sales"].mean().sort_values(ascending=False).reset_index()
fig = px.bar(store_means, x="store", y="sales", title=f"Mean Daily Sales by Store — Item {item}")
st.plotly_chart(fig, width="stretch")

st.subheader("Daily Sales Trend for This Item (All Stores Combined)")
daily = item_df.groupby("date")["sales"].sum().reset_index()
fig2 = px.line(daily, x="date", y="sales", title=f"Total Daily Sales — Item {item}")
st.plotly_chart(fig2, width="stretch")

st.divider()
st.subheader("Model Accuracy for This Item (Validation Set)")
try:
    _, valid, cols = load_feature_tables()
    from src.deployment.inference import get_inference_service

    service = get_inference_service()
    valid_item = valid[valid["item"] == item]
    if len(valid_item) > 0:
        preds = service.model.predict(valid_item[cols])
        valid_item = valid_item.copy()
        valid_item["y_pred"] = preds
        metrics_df = evaluate_by_segment(valid_item, "sales", "y_pred", "item")
        st.dataframe(metrics_df, width="stretch")
    else:
        st.info("No validation rows for this item in the current split.")
except FileNotFoundError:
    st.warning("No trained model found yet — accuracy breakdown unavailable.")

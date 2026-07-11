# Multi-Series Retail Demand Forecasting for Inventory Optimization

An end-to-end forecasting system that predicts the next **90 days** of demand for every
**Store × Item** combination in the Kaggle Store Item Demand Forecasting Challenge
dataset: 5 years of daily sales across **10 stores × 50 items** (913,000 rows, 500 series).

## Results at a Glance

| Model | WAPE | RMSE | MAE | Fit Time |
|---|---|---|---|---|
| **CatBoost** (best) | **10.65%** | 7.56 | 5.82 | 263s |
| XGBoost | 10.69% | 7.57 | 5.84 | 156s |
| LightGBM (tuned) | 10.72% | 7.60 | 5.86 | 88s |
| LightGBM | 10.73% | 7.61 | 5.86 | 127s |
| Random Forest | 11.42% | 8.18 | 6.24 | 269s |
| Seasonal Naive (baseline) | 19.75% | 14.36 | 10.79 | — |
| Moving Average (baseline) | 21.51% | 15.41 | 11.75 | — |
| Naive Last Value (baseline) | 24.29% | 17.65 | 13.27 | — |
| Attention-LSTM* | 33.56% | 22.12 | 18.34 | 169s |

*See `docs/model_documentation.md` for why the LSTM underperforms here — it's a
compute-budget artifact (2 epochs on a single CPU core), not an architectural one.

All four tree-based models beat every baseline by roughly **2x**, and accuracy barely
degrades across the 90-day horizon (day 1 WAPE 11.42% vs. day 90 WAPE 11.58%), confirming
the leak-free feature design works as intended. Full numbers and methodology are in
`docs/results.md`.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the dataset
#    Download train.csv / test.csv from the Kaggle competition and put them under data/raw/

# 3. Run the full pipeline
python -m src.features.pipeline          # feature engineering
python -m src.models.train --model all   # train every model
python -m src.visualization.eda          # EDA charts
python -m src.visualization.time_series_analysis
python -m src.visualization.model_comparison

# 4. Serve
uvicorn api.app:app --reload             # API at http://localhost:8000/docs
streamlit run dashboard/Home.py          # Dashboard at http://localhost:8501
```

See `docs/installation.md` for a complete setup walkthrough, including Docker.

## Project Structure

See `docs/architecture.md` for the full breakdown of every folder and how data flows
through the system.

```
Retail-Demand-Forecasting/
├── data/               # raw + processed (feature) data
├── notebooks/          # exploratory notebooks
├── src/                # all pipeline code (config, data, features, models, evaluation, visualization, deployment, utils)
├── api/                # FastAPI service
├── dashboard/          # Streamlit dashboard (8 pages)
├── tests/              # pytest suite (38 tests)
├── reports/            # generated JSON/CSV metrics reports
├── models/             # trained model artifacts (joblib)
├── images/             # generated charts
├── docs/               # this documentation
└── .github/workflows/  # CI
```

## Documentation

- Installation Guide: `docs/installation.md`
- Architecture: `docs/architecture.md`
- API Documentation: `docs/api_documentation.md`
- Model Documentation: `docs/model_documentation.md`
- Deployment Guide: `docs/deployment_guide.md`
- Results: `docs/results.md`
- Future Work: `docs/future_work.md`

## Key Design Decision: Horizon-Safe Features

Because this project forecasts a full 90-day horizon in one shot ("direct" multi-step
forecasting rather than recursive one-day-at-a-time forecasting), every lag/rolling
feature is computed relative to an **as-of date** of `target_date - 90 days`, not the
naive "yesterday's sales." This is what lets the exact same feature pipeline be used
for training, validation, and true test-set scoring without any leakage — and it's
confirmed empirically: forecast accuracy is nearly flat across the entire 90-day window.
See `src/features/lag_features.py` for the full rationale.

## Compute Environment Note

This project was developed and validated in a sandbox with **1 CPU core**. All reported
metrics are real, measured runs on the actual dataset — nothing here is simulated or
placeholder. Random Forest and Attention-LSTM hyperparameters were scaled down from
their "ideal" defaults (documented in `src/config/settings.py` and
`docs/model_documentation.md`) purely for tractable training time in this environment;
on standard multi-core/GPU infrastructure both would benefit from larger configurations.

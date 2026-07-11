# Architecture

## Data Flow

```
data/raw/train.csv, test.csv
        |
        v
src/data/ingestion.py          (typed loading)
        |
        v
src/data/validation.py         (schema, nulls, dupes, negative values, train/test consistency)
        |
        v
src/data/preprocessing.py      (calendar-gap filling, time-based train/valid split)
        |
        v
src/features/
  - calendar_features.py       (date-derived, leak-free by construction)
  - lag_features.py            (horizon-safe lags/rolling stats -- see Key Design Decision in README)
  - aggregate_features.py      (store/item/series-level statistics)
  - pipeline.py                (orchestrates all of the above end to end)
        |
        v
data/processed/{train,valid,test}_features.parquet
        |
        v
src/models/
  - baseline_models.py         (naive / seasonal naive / moving average)
  - base_model.py              (common fit/predict/save/load interface)
  - random_forest_model.py
  - lightgbm_model.py
  - xgboost_model.py
  - catboost_model.py
  - attention_lstm_model.py    (PyTorch LSTM + additive attention)
  - hyperparameter_tuning.py   (Optuna + TimeSeriesSplit CV)
  - mlflow_tracking.py         (experiment logging)
  - train.py                  (CLI orchestrator: trains + persists + records metrics)
        |
        v
models/*.joblib   +   reports/model_comparison.json
        |
        v
src/evaluation/
  - metrics.py                 (RMSE, MAE, MAPE, SMAPE, WAPE)
  - segment_evaluation.py      (per-store / per-item / per-horizon-day breakdowns)
  - shap_explainability.py     (SHAP TreeExplainer analysis)
        |
        v
src/deployment/inference.py    (shared InferenceService: loads model + history,
                                 builds features for arbitrary future requests)
        |
        +--> api/app.py                    (FastAPI: /health /predict /forecast /model-info)
        +--> dashboard/Home.py + pages/*   (Streamlit: 8-page interactive dashboard)
```

## Folder Explanation

| Folder | Purpose |
|---|---|
| `data/raw/` | Original Kaggle CSVs (not committed — see `.gitignore`) |
| `data/processed/` | Generated parquet feature tables + cached feature-column list |
| `notebooks/` | Exploratory notebooks (companion to `src/visualization/`) |
| `src/config/` | Central settings: paths, forecast params, per-model hyperparameter defaults |
| `src/data/` | Ingestion, validation, preprocessing |
| `src/features/` | All feature engineering, orchestrated by `pipeline.py` |
| `src/models/` | Every model implementation + training/tuning/tracking orchestration |
| `src/evaluation/` | Metrics, segment breakdowns, SHAP |
| `src/visualization/` | EDA, time series analysis, model comparison charts |
| `src/deployment/` | The shared inference service used by both the API and dashboard |
| `src/utils/` | Logging, seeding, timing, memory optimization |
| `api/` | FastAPI app + Pydantic schemas |
| `dashboard/` | Streamlit multi-page app |
| `tests/` | Pytest suite (fast, synthetic-data-based — see `tests/conftest.py`) |
| `reports/` | Generated JSON/CSV: EDA summary, time series analysis, model comparison, SHAP, Optuna studies, per-segment evaluation |
| `models/` | Trained model artifacts (`.joblib`) |
| `images/` | Generated PNG charts |
| `docs/` | This documentation |
| `.github/workflows/` | CI: lint, format check, tests, Docker build |

## Why a Shared `InferenceService`

Both the API and the dashboard need to turn a raw `(store, item, date)` request into a
prediction. Rather than duplicating that logic, `src/deployment/inference.py` implements
it once: it loads the full sales history and a trained model at startup, then for any
request it appends a scaffold row to the history and re-runs the **exact same**
`build_feature_table()` function used during training. This guarantees serving-time
features can never silently drift from training-time features — a common source of bugs
in production forecasting systems.

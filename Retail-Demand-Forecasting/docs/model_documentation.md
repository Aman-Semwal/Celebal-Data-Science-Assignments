# Model Documentation

## Task Framing

Every model in this project solves the same task: **direct 90-day-ahead point
forecasting** of daily units sold for a given (store, item) pair. "Direct" means the
model predicts every day of the horizon from features computed as of a single fixed
as-of date (`target_date - 90 days`) — it does **not** forecast day 1, feed that back in,
forecast day 2, etc. (See the README's "Key Design Decision" section and
`src/features/lag_features.py` for why.) This keeps every model directly comparable and
avoids recursive-forecasting error accumulation.

## Baselines (`src/models/baseline_models.py`)

| Model | Logic | WAPE |
|---|---|---|
| Naive Last Value | Repeats the last known value as of the as-of date | 24.29% |
| Seasonal Naive | Repeats the value from exactly 365 days before the target | 19.75% |
| Moving Average | Mean of the trailing 28 days as of the as-of date | 21.51% |

Seasonal Naive beating the other two makes sense given the strong yearly seasonality
found in the STL decomposition (see `docs/results.md`).

## Random Forest (`src/models/random_forest_model.py`)

Scikit-learn `RandomForestRegressor`. Categorical columns (store, item) are ordinal-encoded
since sklearn doesn't support pandas categoricals natively. **Compute-constrained
defaults**: `n_estimators=60, max_depth=10, max_samples=0.25` — tuned down from a more
typical `n_estimators=300, max_depth=14` specifically because this project was validated
on a single-CPU-core sandbox (RF doesn't benefit from LightGBM/XGBoost/CatBoost's
histogram-based training speed). On multi-core hardware, increase these back up.

## LightGBM / XGBoost / CatBoost (gradient boosting)

All three use native categorical feature support for `store`/`item` (avoiding one-hot
blowup), early stopping against the validation set, and the same underlying feature set.
They finished within half a percentage point of each other on WAPE (10.65%-10.73%),
which is a common finding when tree-based GBM libraries are given the same features and
comparable hyperparameter budgets — the real driver of that class's superiority over
baselines is the feature engineering, not the specific implementation.

CatBoost edged out the others slightly and had the best fit-time/accuracy trade-off among
the untuned models.

## Attention-LSTM (`src/models/attention_lstm_model.py`)

**Architecture** (fully implemented, not a stub): an embedding layer for store/item ids,
a multi-layer LSTM over the trailing `sequence_length` days of sales (ending at the as-of
date), a Bahdanau-style additive attention layer that pools the LSTM's per-timestep
outputs into a single context vector, and a small feed-forward head combining that
context with the store/item embeddings to output one prediction.

**Honest performance note**: the persisted model in this repo was trained for only
**2 epochs** on a **2-year subsample** of the data, on a **single CPU core**, because a
full run (recommended defaults: `sequence_length=90, hidden_size=64, epochs=15` on the
full 5-year history) would take well over an hour on this hardware. The result (33.6%
WAPE) is worse than even the naive baselines — this is a training-budget artifact, not
evidence the architecture doesn't work. Deep sequence models generally need either many
more epochs or a GPU to reach competitive performance on tabular-adjacent tasks like this
one, where gradient-boosted trees already have a strong inductive-bias advantage. To
reproduce with a realistic budget:

```python
from src.models.attention_lstm_model import AttentionLSTMModel
model = AttentionLSTMModel(sequence_length=90, hidden_size=64, num_layers=2, epochs=15)
model.fit(full_history_df, train_target_df, valid_target_df)
```

## Hyperparameter Tuning (`src/models/hyperparameter_tuning.py`)

Optuna with `TimeSeriesSplit` cross-validation (never validates on a fold that precedes
its training fold in time). Demonstrated end-to-end on LightGBM: 4 trials, 2 folds, on a
25%-of-history subsample (again, a concession to single-core compute) — applying the
resulting best parameters to a full-data retrain improved WAPE marginally (10.734% ->
10.721%). With a full trial budget (`n_trials=20-50`) on complete data, expect a larger
gain; the machinery scales directly, only `n_trials`/`sample_frac` need to change.

## SHAP Explainability (`src/evaluation/shap_explainability.py`)

Uses SHAP's `TreeExplainer` (exact for tree ensembles, no approximation) on a sample of
validation rows. For CatBoost, the top drivers were `month_cos`/`month_sin` (yearly
seasonality), `store_item_mean_sales`/`store_item_std_sales` (series-level baseline), and
`day_of_week` — consistent with what the EDA and STL decomposition show.

## MLflow Tracking (`src/models/mlflow_tracking.py`)

Every run wrapped in `mlflow.start_run()` logs: model name, feature count, train/valid
row counts, the horizon, all hyperparameters, all validation metrics, fit time, and the
serialized model artifact. Backend is a local SQLite database (`mlflow.db`) rather than
the deprecated raw-filesystem store. View with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

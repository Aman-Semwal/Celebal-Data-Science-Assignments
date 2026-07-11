# Results

## Dataset

- 913,000 rows, 10 stores x 50 items = 500 series, daily granularity, 2013-01-01 to 2017-12-31
- Mean daily sales: 52.25 units, std: 28.80, median: 47, range: 0-231
- No calendar gaps (every series is fully continuous)
- Test set: exactly 90 days (2018-01-01 to 2018-03-31) per series, matching the project's
  forecast horizon

## Exploratory Findings

- **Trend**: a clear, steady upward trend across all 5 years (STL decomposition, see
  `images/stl_aggregate.png`)
- **Seasonality**: strong yearly seasonality (summer peak, winter trough) and weekly
  seasonality (weekends higher than weekdays)
- **Stationarity**: the ADF test rejects the null of a unit root for both the aggregate
  series and an example single (store, item) series (see `reports/time_series_analysis.json`),
  i.e. both are statistically stationary around their trend/seasonal components — this
  supports using lag/rolling features directly rather than differencing

## Model Comparison (Validation Set, 90-Day Held-Out Window)

| Model | WAPE | RMSE | MAE | MAPE | SMAPE | Fit Time |
|---|---|---|---|---|---|---|
| CatBoost | 10.65% | 7.56 | 5.82 | 12.78% | 12.27% | 263s |
| XGBoost | 10.69% | 7.57 | 5.84 | 12.87% | 12.30% | 156s |
| LightGBM (Optuna-tuned) | 10.72% | 7.60 | 5.86 | 12.95% | 12.35% | 88s |
| LightGBM | 10.73% | 7.61 | 5.86 | 12.90% | 12.34% | 127s |
| Random Forest | 11.42% | 8.18 | 6.24 | 13.39% | 12.95% | 269s |
| Seasonal Naive | 19.75% | 14.36 | 10.79 | -- | -- | 0s |
| Moving Average | 21.51% | 15.41 | 11.75 | -- | -- | 0s |
| Naive Last Value | 24.29% | 17.65 | 13.27 | -- | -- | 0s |
| Attention-LSTM | 33.56% | 22.12 | 18.34 | -- | -- | 169s |

(Attention-LSTM trained for only 2 epochs on a 2-year subsample due to single-CPU-core
sandbox constraints — see `docs/model_documentation.md` for the full explanation.)

## Segment-Level Analysis (CatBoost)

| Segment | Best | Worst |
|---|---|---|
| Store | Store 2 (WAPE 9.54%) | Store 7 (WAPE 12.67%) |
| Item | Item 15 (WAPE 8.31%) | Item 5 (WAPE 18.04%) |

Item 5's much higher error suggests it's a genuinely harder-to-predict, likely
lower-volume or more volatile SKU — worth a dedicated look before relying on its forecast
for automated reordering.

## Horizon Degradation Check

| Horizon Day | WAPE |
|---|---|
| Day 1 (nearest to as-of date) | 11.42% |
| Day 90 (furthest from as-of date) | 11.58% |

The near-flat degradation (+0.16 percentage points across the full 90-day window)
confirms the horizon-safe feature design (see README) is working as intended — there's
no cliff in accuracy as you look further into the forecast, because every day in the
horizon is predicted from features anchored to the same as-of date, not accumulated
recursive error.

## Hyperparameter Tuning

Optuna + TimeSeriesSplit (2 folds) on a 25%-of-history subsample of LightGBM, 4 trials,
found a configuration with in-CV WAPE 20.17% (not directly comparable to the full-data
numbers above, since it's cross-validated on a much smaller, coarser split). Retraining
LightGBM on the full dataset with those tuned parameters improved held-out WAPE from
10.734% to 10.721% — a modest but real gain given the very limited trial budget used
here for tractability; a full budget (e.g. `n_trials=30+` on complete data) would likely
yield more.

## SHAP Feature Importance (CatBoost, top 5 by mean |SHAP value|)

1. `month_cos` — yearly seasonality (cyclical encoding)
2. `store_item_mean_sales` — series-level baseline volume
3. `store_item_std_sales` — series-level volatility
4. `year` — captures the overall upward trend
5. `month_sin` — yearly seasonality (cyclical encoding)

This lines up exactly with the EDA/STL findings: seasonality and each series' own
baseline level dominate, with the boosted trees learning fine adjustments on top.

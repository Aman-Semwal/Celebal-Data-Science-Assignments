# Future Work

## Compute-Constrained Items (Highest Priority If More Compute Is Available)

- **Random Forest**: retrain with the "standard" defaults noted in
  `docs/model_documentation.md` (`n_estimators=300, max_depth=14`) on multi-core hardware.
- **Attention-LSTM**: retrain for the full recommended budget
  (`sequence_length=90, hidden_size=64, epochs=15`, full 5-year history) on a GPU or
  multi-core machine. Also worth trying: teacher forcing with a recursive rollout for
  comparison against the direct-forecast framing used here, and a simple
  Transformer/temporal-fusion-transformer variant as a stronger deep-learning baseline.
- **Optuna tuning**: run with a full trial budget (30-100 trials) on the complete
  dataset rather than a 25% subsample, and extend tuning to XGBoost, CatBoost, and
  Random Forest (the search spaces are already implemented in
  `src/models/hyperparameter_tuning.py` — only the invocation needs scaling up).

## Modeling

- **Ensembling**: a simple weighted average or stacked meta-learner over
  CatBoost/XGBoost/LightGBM would likely beat any single model, given how close their
  individual errors are but how different their individual mistakes probably look.
- **Quantile forecasts**: inventory decisions care about the distribution of demand,
  not just the point estimate — LightGBM/CatBoost both support quantile loss functions
  natively, which would let the dashboard show P10/P50/P90 forecast bands directly
  relevant to safety-stock calculations.
- **Recursive vs. direct comparison**: this project uses a direct 90-day-ahead framing
  throughout for leakage-safety and simplicity; a recursive (day-by-day, feeding
  predictions back in) approach with proper error-accumulation analysis would be a
  valuable comparison.
- **External regressors**: promotions, price changes, and holidays are common demand
  drivers not present in this dataset; if available, they'd likely meaningfully improve
  accuracy on top of the current calendar/lag/aggregate feature set.

## Engineering

- **Feature store**: as more series or a longer horizon get added, moving lag/rolling
  computation into a proper feature store (e.g. incremental updates rather than full
  recomputation) would improve pipeline latency.
- **Model registry**: MLflow tracking is wired up (`src/models/mlflow_tracking.py`); the
  natural next step is MLflow's Model Registry for versioned promotion (staging ->
  production) instead of the current file-naming-convention-based `_MODEL_PREFERENCE`
  list in `src/deployment/inference.py`.
- **Monitoring**: add drift detection on the feature distributions and rolling-window
  accuracy tracking in production, alerting if WAPE on recent actuals exceeds the
  validation-set baseline by a meaningful margin.
- **Batch scoring**: the dashboard's "Download Forecast" batch mode loops one
  (store, item) series at a time; for very large catalogs this should become a proper
  batch job (e.g. via a queue) rather than a synchronous in-app loop.

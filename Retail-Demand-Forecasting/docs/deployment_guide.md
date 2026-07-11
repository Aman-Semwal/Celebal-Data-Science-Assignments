# Deployment Guide

## Local Development

```bash
uvicorn api.app:app --reload --port 8000
streamlit run dashboard/Home.py --server.port 8501
```

## Docker

Build:
```bash
docker build -t retail-demand-forecasting .
```

Run the API:
```bash
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  retail-demand-forecasting
```

Run the dashboard (override the default CMD):
```bash
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  retail-demand-forecasting \
  streamlit run dashboard/Home.py --server.address 0.0.0.0 --server.port 8501
```

Mounting `data/`, `models/`, and `reports/` as volumes means you can retrain outside the
container and the service picks up new artifacts on restart, without rebuilding the
image.

## Production Considerations

- **Model loading**: `InferenceService` (see `src/deployment/inference.py`) loads the
  full sales history and model once per process via `functools.lru_cache`. In a
  multi-worker deployment (e.g. `uvicorn --workers 4`), each worker loads its own copy —
  size your container memory accordingly (the full history DataFrame is small, tens of
  MB, so this is not a major concern at this dataset's scale).
- **Retraining cadence**: this is a retail demand problem with strong yearly seasonality
  and a real (if modest) upward trend — retrain at least monthly, and any time a new full
  season of data becomes available, to keep the lag/rolling features current.
- **Model promotion**: `_MODEL_PREFERENCE` in `src/deployment/inference.py` controls
  which trained model gets served. To promote a newly retrained model, either overwrite
  the highest-preference `.joblib` file or reorder the preference list, then restart the
  service (or clear the `lru_cache`).
- **Horizon boundary**: the API's `/forecast` endpoint always forecasts starting the day
  after the last date in the training history. If you retrain on fresher data, the
  forecast window advances automatically.
- **Health checks**: point your orchestrator's liveness/readiness probe at `GET /health`.
  It returns `200` even when degraded (no model loaded) so it won't falsely restart a
  container that's simply waiting for a model to be mounted — check the `status` field
  in the response body, not just the HTTP status code, if you need to gate readiness on a
  loaded model.
- **CI/CD**: `.github/workflows/ci.yml` lints (ruff), format-checks (black), runs the
  full pytest suite, and verifies the Docker image builds on every push/PR to `main`.

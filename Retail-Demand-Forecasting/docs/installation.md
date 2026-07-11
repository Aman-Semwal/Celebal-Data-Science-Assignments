# Installation Guide

## Prerequisites

- Python 3.12+
- ~2 GB free disk space (for dependencies + trained models)
- The dataset: `train.csv` and `test.csv` from the Kaggle
  Store Item Demand Forecasting Challenge

## Option A: Local (pip)

```bash
git clone <this-repo-url>
cd Retail-Demand-Forecasting

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .                   # installs src/ and api/ as importable packages
```

Place the dataset:

```bash
mkdir -p data/raw
cp /path/to/train.csv /path/to/test.csv data/raw/
```

Run the pipeline:

```bash
python -m src.features.pipeline
python -m src.models.train --model all
```

Launch the services:

```bash
uvicorn api.app:app --reload           # http://localhost:8000/docs
streamlit run dashboard/Home.py        # http://localhost:8501
```

## Option B: Docker

```bash
docker build -t retail-demand-forecasting .
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models retail-demand-forecasting
```

The container's default `CMD` starts the FastAPI service. To run the dashboard instead:

```bash
docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
  retail-demand-forecasting streamlit run dashboard/Home.py --server.address 0.0.0.0
```

## Verifying the Install

```bash
pytest tests/ -v          # should show 38 passed
ruff check src api dashboard tests
black --check src api dashboard tests
```

## Common Issues

- **`FileNotFoundError: Training data not found`** — you haven't placed `train.csv`
  under `data/raw/` yet.
- **`No trained model found`** (API/dashboard) — run
  `python -m src.models.train --model catboost` (or `all`) first.
- **Slow training on constrained hardware** — the default hyperparameters in
  `src/config/settings.py` were tuned to be tractable on a single CPU core. If you have
  more cores/a GPU available, increase `n_estimators`, `max_depth`, and (for the
  Attention-LSTM) `epochs` for better accuracy.
- **MLflow "filesystem tracking backend... maintenance mode"** — this project uses a
  SQLite backend (`mlflow.db`) by default specifically to avoid this; if you see it,
  check `MLFLOW_TRACKING_URI` in `src/config/settings.py` hasn't been changed to a
  bare `file:` URI.

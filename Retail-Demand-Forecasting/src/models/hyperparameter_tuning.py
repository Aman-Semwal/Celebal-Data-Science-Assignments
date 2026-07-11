"""Hyperparameter optimization via Optuna, using leakage-safe TimeSeriesSplit CV.

Supports tuning any of the tree-based models (LightGBM, XGBoost, CatBoost,
RandomForest). Each trial fits the model on each of ``n_splits`` expanding
time-series folds and reports the mean WAPE across folds, so Optuna never
sees a validation fold that precedes its training fold in time.
"""

from __future__ import annotations

import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config.settings import FORECAST, PATHS
from src.evaluation.metrics import wape
from src.models.catboost_model import CatBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.utils.common import save_json
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

_MODEL_CLASSES = {
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
    "random_forest": RandomForestModel,
}


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    """Return a hyperparameter dict sampled from the search space for ``model_name``."""
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }
    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_seed": 42,
            "verbose": False,
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 40, 200, step=20),
            "max_depth": trial.suggest_int("max_depth", 6, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "max_samples": trial.suggest_float("max_samples", 0.2, 0.7),
            "n_jobs": -1,
            "random_state": 42,
        }
    raise ValueError(f"No search space defined for model '{model_name}'.")


def make_objective(
    model_name: str, df: pd.DataFrame, feature_columns: list[str], n_splits: int = 3
):
    """Build an Optuna objective that runs TimeSeriesSplit CV for ``model_name``.

    Args:
        model_name: One of the keys in ``_MODEL_CLASSES``.
        df: Feature-complete DataFrame sorted by date, used for CV folds.
        feature_columns: Which columns of ``df`` are model features.
        n_splits: Number of expanding-window TimeSeriesSplit folds.

    Returns:
        A callable ``objective(trial) -> float`` (mean WAPE, minimized).
    """
    model_cls = _MODEL_CLASSES[model_name]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    df_sorted = df.sort_values(FORECAST.date_col).reset_index(drop=True)
    X = df_sorted[feature_columns]
    y = df_sorted[FORECAST.target_col]

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_name)
        fold_scores = []
        for train_idx, valid_idx in tscv.split(X):
            model = model_cls(**params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[valid_idx])
            fold_scores.append(wape(y.iloc[valid_idx].to_numpy(), preds))
        return sum(fold_scores) / len(fold_scores)

    return objective


def run_optuna_tuning(
    model_name: str, n_trials: int = 20, n_splits: int = 3, sample_frac: float = 1.0
) -> dict:
    """Run an Optuna study for ``model_name`` and persist the best params + study summary.

    Args:
        model_name: Which model to tune.
        n_trials: Number of Optuna trials.
        n_splits: TimeSeriesSplit fold count.
        sample_frac: Optionally subsample rows (by most-recent date) to
            keep tuning tractable on constrained compute; 1.0 uses all data.

    Returns:
        Dict with best_params, best_value (mean WAPE), and per-trial history.
    """
    train_df = pd.read_parquet(PATHS.train_features_parquet)
    feature_columns = [
        c
        for c in train_df.columns
        if c
        not in (FORECAST.date_col, FORECAST.store_col, FORECAST.item_col, FORECAST.target_col, "id")
    ]

    if sample_frac < 1.0:
        cutoff = train_df[FORECAST.date_col].max() - pd.Timedelta(
            days=int(
                (train_df[FORECAST.date_col].max() - train_df[FORECAST.date_col].min()).days
                * sample_frac
            )
        )
        train_df = train_df[train_df[FORECAST.date_col] >= cutoff]
        logger.info(
            "Subsampled tuning data to %d rows (sample_frac=%.2f) for tractability.",
            len(train_df),
            sample_frac,
        )

    objective = make_objective(model_name, train_df, feature_columns, n_splits=n_splits)

    study = optuna.create_study(direction="minimize", study_name=f"{model_name}_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    result = {
        "model_name": model_name,
        "best_params": study.best_params,
        "best_value_wape": study.best_value,
        "n_trials": n_trials,
        "n_splits": n_splits,
        "trials": [
            {"number": t.number, "value": t.value, "params": t.params} for t in study.trials
        ],
    }
    save_json(result, PATHS.reports_dir / f"optuna_{model_name}.json")
    logger.info("Best %s params (WAPE=%.3f): %s", model_name, study.best_value, study.best_params)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lightgbm", choices=list(_MODEL_CLASSES))
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--sample-frac", type=float, default=1.0)
    args = parser.parse_args()

    run_optuna_tuning(
        args.model, n_trials=args.n_trials, n_splits=args.n_splits, sample_frac=args.sample_frac
    )

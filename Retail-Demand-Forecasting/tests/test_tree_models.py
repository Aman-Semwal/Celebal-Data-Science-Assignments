"""Tests for the tree-based model wrappers (LightGBM, XGBoost, CatBoost, RandomForest).

Uses tiny hyperparameters on the synthetic dataset so these tests run in
well under a second each, independent of the real 913K-row dataset.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.features.calendar_features import add_calendar_features
from src.features.lag_features import add_lag_and_rolling_features
from src.models.catboost_model import CatBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel

_FAST_PARAMS = {
    "lightgbm": {"n_estimators": 10, "num_leaves": 7, "verbosity": -1},
    "xgboost": {"n_estimators": 10, "max_depth": 3},
    "catboost": {"iterations": 10, "depth": 3, "verbose": False},
    "random_forest": {"n_estimators": 5, "max_depth": 3, "max_samples": 0.8},
}


def _make_features(df):
    out = add_calendar_features(df)
    out = add_lag_and_rolling_features(out, lag_days=[1, 7], rolling_windows=[7], horizon_days=10)
    out["store"] = out["store"].astype("category")
    out["item"] = out["item"].astype("category")
    return out.dropna()


@pytest.mark.parametrize(
    "model_cls,param_key",
    [
        (LightGBMModel, "lightgbm"),
        (XGBoostModel, "xgboost"),
        (CatBoostModel, "catboost"),
        (RandomForestModel, "random_forest"),
    ],
)
def test_model_fit_predict_roundtrip(synthetic_train_df, tmp_path, model_cls, param_key):
    features = _make_features(synthetic_train_df)
    feature_cols = [c for c in features.columns if c not in ("date", "sales")]
    X, y = features[feature_cols], features["sales"]

    model = model_cls(**_FAST_PARAMS[param_key])
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert np.all(preds >= 0)  # non-negativity clipping
    assert np.all(np.isfinite(preds))

    save_path = tmp_path / f"{model.name}.joblib"
    model.save(save_path)
    assert save_path.exists()

    reloaded = model_cls()
    reloaded.load(save_path)
    reloaded_preds = reloaded.predict(X)
    np.testing.assert_allclose(preds, reloaded_preds, rtol=1e-5)

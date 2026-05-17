"""Tests for Step 5 XGBoost factor modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation import evaluate_factor_predictions, temporal_train_test_split
from models import XGBoostFactorModel, XGBoostModelConfig
from optimization import tune_xgboost_factor_model


def _factor_dataset(rows: int = 140) -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B", tz="UTC")
    trend = np.linspace(-1.0, 1.0, rows)
    cycle = np.sin(np.linspace(0.0, 10.0, rows))
    volatility = np.cos(np.linspace(0.0, 6.0, rows))
    target = 0.04 * trend + 0.02 * cycle - 0.01 * volatility

    features = pd.DataFrame(
        {
            "momentum_factor_10": trend,
            "macd_histogram": cycle,
            "rolling_volatility_20": volatility,
            "volume_zscore_20": np.sin(np.linspace(0.0, 3.0, rows)),
        },
        index=dates,
    )
    return features, pd.Series(target, index=dates, name="forward_return_1")


def test_xgboost_regression_factor_model_fits_and_predicts() -> None:
    features, target = _factor_dataset()
    split = temporal_train_test_split(features, target, test_size=20)
    config = XGBoostModelConfig(
        objective_type="regression",
        n_estimators=40,
        max_depth=2,
        learning_rate=0.1,
    )

    model = XGBoostFactorModel(config).fit(split.x_train, split.y_train)
    prediction = model.predict(split.x_test)

    assert prediction.model_name == "xgboost_factor"
    assert prediction.predictions.index.equals(split.x_test.index)
    assert model.feature_importance.index.isin(features.columns).all()


def test_xgboost_classification_factor_model_predicts_probabilities() -> None:
    features, returns = _factor_dataset()
    direction = (returns > 0.0).astype(int)
    split = temporal_train_test_split(features, direction, test_size=20)
    config = XGBoostModelConfig(
        objective_type="classification",
        n_estimators=30,
        max_depth=2,
        learning_rate=0.1,
    )

    model = XGBoostFactorModel(config).fit(split.x_train, split.y_train)
    predictions = model.predict(split.x_test).predictions

    assert predictions.between(0.0, 1.0).all()


def test_factor_prediction_report_contains_ml_and_trading_metrics() -> None:
    features, target = _factor_dataset()
    split = temporal_train_test_split(features, target, test_size=20)
    model = XGBoostFactorModel(
        XGBoostModelConfig(n_estimators=40, max_depth=2, learning_rate=0.1)
    ).fit(split.x_train, split.y_train)

    predictions = model.predict(split.x_test).predictions
    report = evaluate_factor_predictions(predictions, split.y_test)

    assert report.observations == 20
    assert report.rmse >= 0.0
    assert 0.0 <= report.accuracy <= 1.0
    assert report.max_drawdown <= 0.0
    assert -1.0 <= report.information_coefficient <= 1.0


def test_temporal_xgboost_tuning_selects_candidate_config() -> None:
    features, target = _factor_dataset()
    candidates = [
        XGBoostModelConfig(n_estimators=10, max_depth=1, learning_rate=0.05),
        XGBoostModelConfig(n_estimators=30, max_depth=2, learning_rate=0.1),
    ]

    result = tune_xgboost_factor_model(
        features,
        target,
        candidate_configs=candidates,
        test_size=20,
    )

    assert result.config in candidates
    assert result.evaluation.observations == 20

"""Tests for Step 6 LSTM time-series modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation import evaluate_factor_predictions, temporal_train_test_split
from models import LSTMForecaster, LSTMModelConfig, create_sequence_dataset


def _sequential_dataset(rows: int = 90) -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B", tz="UTC")
    signal = np.sin(np.linspace(0.0, 12.0, rows))
    trend = np.linspace(-0.5, 0.5, rows)
    volatility = np.cos(np.linspace(0.0, 4.0, rows))
    target = 0.03 * signal + 0.015 * trend - 0.005 * volatility

    features = pd.DataFrame(
        {
            "log_return_1": signal,
            "momentum_factor_10": trend,
            "rolling_volatility_20": volatility,
        },
        index=dates,
    )
    return features, pd.Series(target, index=dates, name="forward_return_1")


def test_sequence_dataset_uses_sliding_windows_with_prediction_timestamps() -> None:
    features, target = _sequential_dataset(rows=12)
    dataset = create_sequence_dataset(features, target, sequence_length=5)

    assert dataset.x.shape == (8, 5, 3)
    assert dataset.y.shape == (8,)
    assert dataset.index[0] == features.index[4]
    assert dataset.feature_names == list(features.columns)


def test_lstm_regression_forecaster_trains_and_predicts() -> None:
    features, target = _sequential_dataset()
    split = temporal_train_test_split(features, target, test_size=30)
    config = LSTMModelConfig(
        sequence_length=8,
        hidden_size=8,
        epochs=4,
        batch_size=16,
        learning_rate=0.01,
        objective_type="regression",
    )

    model = LSTMForecaster(config).fit(split.x_train, split.y_train)
    predictions = model.predict(split.x_test).predictions
    aligned_target = split.y_test.loc[predictions.index]
    report = evaluate_factor_predictions(predictions, aligned_target)

    assert len(model.history.losses) == config.epochs
    assert predictions.index.min() == split.x_test.index[config.sequence_length - 1]
    assert report.observations == len(predictions)
    assert report.rmse >= 0.0


def test_lstm_classification_forecaster_outputs_probabilities() -> None:
    features, returns = _sequential_dataset()
    target = (returns > 0.0).astype(float)
    split = temporal_train_test_split(features, target, test_size=30)
    config = LSTMModelConfig(
        sequence_length=8,
        hidden_size=8,
        epochs=3,
        batch_size=16,
        learning_rate=0.01,
        objective_type="classification",
    )

    model = LSTMForecaster(config).fit(split.x_train, split.y_train)
    predictions = model.predict(split.x_test).predictions

    assert predictions.between(0.0, 1.0).all()
    assert predictions.index.equals(split.x_test.index[config.sequence_length - 1 :])


def test_lstm_predict_dataset_accepts_pre_windowed_data() -> None:
    features, target = _sequential_dataset()
    config = LSTMModelConfig(sequence_length=6, hidden_size=6, epochs=2, learning_rate=0.01)
    model = LSTMForecaster(config).fit(features.iloc[:60], target.iloc[:60])
    dataset = create_sequence_dataset(features.iloc[60:], target.iloc[60:], sequence_length=6)

    predictions = model.predict_dataset(dataset)

    assert predictions.index.equals(dataset.index)
    assert predictions.name == "regression_prediction"

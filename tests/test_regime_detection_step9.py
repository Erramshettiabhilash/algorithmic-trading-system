"""Tests for Step 9 regime detection and regime-aware modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.research import ModelPrediction
from regime import (
    ClusteringRegimeDetector,
    HMMRegimeDetector,
    RegimeAwareModel,
    RegimeDetectionConfig,
    RegimeModelConfig,
    classify_market_regimes,
    create_regime_feature_matrix,
)


def _price_phases(rows_per_phase: int = 45) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ranging = 100.0 + np.sin(np.linspace(0.0, 8.0, rows_per_phase))
    trending = ranging[-1] + np.linspace(0.0, 18.0, rows_per_phase)
    high_volatility = trending[-1] + np.cumsum(rng.normal(0.0, 1.8, rows_per_phase))
    close = np.concatenate([ranging, trending, high_volatility])
    dates = pd.date_range("2024-01-01", periods=len(close), freq="B", tz="UTC")

    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": 1_000_000 + 1_000 * np.arange(len(close)),
        },
        index=dates,
    )


def _model_features(index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    n_rows = len(index)
    x = pd.DataFrame(
        {
            "momentum": np.linspace(-1.0, 1.0, n_rows),
            "volatility": np.cos(np.linspace(0.0, 8.0, n_rows)),
        },
        index=index,
    )
    regimes = pd.Series(
        np.where(np.arange(n_rows) < n_rows // 2, "trending", "ranging"),
        index=index,
        name="regime",
    )
    y = pd.Series(
        np.where(regimes == "trending", 0.03 * x["momentum"], -0.02 * x["momentum"]),
        index=index,
        name="forward_return_1",
    )
    return x, y, regimes


@dataclass
class _MeanByFeatureModel:
    coefficient: float = 0.0
    target_name: str = "target"

    def fit(self, features: pd.DataFrame, target: pd.Series) -> _MeanByFeatureModel:
        self.coefficient = float(target.mean())
        self.target_name = target.name or self.target_name
        return self

    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        predictions = pd.Series(self.coefficient, index=features.index, name="prediction")
        return ModelPrediction(
            predictions=predictions,
            target_name=self.target_name,
            model_name="mean_by_regime",
        )


def test_create_regime_feature_matrix_contains_core_diagnostics() -> None:
    prices = _price_phases()
    features = create_regime_feature_matrix(
        prices,
        RegimeDetectionConfig(return_window=10, volatility_window=10, trend_window=10),
    )

    assert {"rolling_return", "rolling_volatility", "trend_strength"}.issubset(features.columns)
    assert features.index.is_monotonic_increasing
    assert features.isna().sum().sum() == 0


def test_deterministic_market_regime_classifier_labels_known_states() -> None:
    prices = _price_phases()
    regimes = classify_market_regimes(
        prices,
        RegimeDetectionConfig(return_window=10, volatility_window=10),
    )

    assert regimes.isin({"ranging", "trending", "high_volatility"}).all()
    assert "high_volatility" in set(regimes.tail(30))


def test_clustering_regime_detector_predicts_readable_labels() -> None:
    prices = _price_phases()
    matrix = create_regime_feature_matrix(prices, RegimeDetectionConfig(return_window=10))
    detector = ClusteringRegimeDetector(RegimeDetectionConfig(n_regimes=3)).fit(matrix)

    regimes = detector.predict(matrix.tail(20))

    assert regimes.index.equals(matrix.tail(20).index)
    assert regimes.isin({"ranging", "trending", "high_volatility"}).all()


def test_hmm_regime_detector_predicts_readable_labels() -> None:
    prices = _price_phases()
    matrix = create_regime_feature_matrix(prices, RegimeDetectionConfig(return_window=10))
    detector = HMMRegimeDetector(RegimeDetectionConfig(n_regimes=3, random_state=3)).fit(matrix)

    regimes = detector.predict(matrix.tail(20))

    assert regimes.index.equals(matrix.tail(20).index)
    assert regimes.isin({"ranging", "trending", "high_volatility"}).all()


def test_regime_aware_model_trains_and_switches_between_models() -> None:
    index = pd.date_range("2024-01-01", periods=80, freq="B", tz="UTC")
    features, target, regimes = _model_features(index)
    model = RegimeAwareModel(
        model_factory=_MeanByFeatureModel,
        config=RegimeModelConfig(min_regime_observations=10),
    ).fit(features, target, regimes)

    prediction = model.predict(features.tail(20), regimes.tail(20))

    assert prediction.model_name == "regime_aware_model"
    assert prediction.predictions.index.equals(features.tail(20).index)
    assert {"fallback", "trending", "ranging"}.issubset(model.models)

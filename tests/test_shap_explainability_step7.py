"""Tests for Step 7 SHAP explainability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from explainability import ShapExplainabilityEngine, interpret_feature_effect
from models import XGBoostFactorModel, XGBoostModelConfig


def _factor_dataset(rows: int = 80) -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B", tz="UTC")
    momentum = np.linspace(-1.0, 1.0, rows)
    volatility = np.cos(np.linspace(0.0, 6.0, rows))
    volume = np.sin(np.linspace(0.0, 5.0, rows))
    target = 0.05 * momentum - 0.02 * volatility + 0.01 * volume

    features = pd.DataFrame(
        {
            "momentum_factor_10": momentum,
            "rolling_volatility_20": volatility,
            "volume_zscore_20": volume,
        },
        index=dates,
    )
    return features, pd.Series(target, index=dates, name="forward_return_1")


def _fit_model() -> tuple[XGBoostFactorModel, pd.DataFrame]:
    features, target = _factor_dataset()
    model = XGBoostFactorModel(
        XGBoostModelConfig(
            n_estimators=20,
            max_depth=2,
            learning_rate=0.1,
            random_state=7,
        )
    ).fit(features, target)
    return model, features


def test_shap_engine_returns_timestamp_aligned_values() -> None:
    model, features = _fit_model()
    sample = features.tail(20)
    explanation = ShapExplainabilityEngine(model).explain(sample)

    assert explanation.values.shape == sample.shape
    assert explanation.values.index.equals(sample.index)
    assert explanation.values.columns.equals(sample.columns)
    assert isinstance(explanation.base_value, float)


def test_shap_feature_importance_and_dependence_data() -> None:
    model, features = _fit_model()
    engine = ShapExplainabilityEngine(model)
    explanation = engine.explain(features.tail(20))

    importance = engine.feature_importance(explanation)
    dependence = engine.dependence_data(explanation, "momentum_factor_10")

    assert importance.index.isin(features.columns).all()
    assert importance.iloc[0] >= 0.0
    assert list(dependence.columns) == ["feature_value", "shap_value"]
    assert dependence.index.equals(explanation.data.index)


def test_local_explanation_reconciles_to_shap_prediction_value() -> None:
    model, features = _fit_model()
    engine = ShapExplainabilityEngine(model)
    explanation = engine.explain(features.tail(20))

    local = engine.local_explanation(explanation, row=-1, top_n=2)
    full_prediction = explanation.base_value + explanation.values.iloc[-1].sum()

    assert local.timestamp == explanation.values.index[-1]
    assert np.isclose(local.prediction_value, full_prediction)
    assert len(local.contributions) == 2


def test_interaction_summary_and_feature_interpretation_are_readable() -> None:
    model, features = _fit_model()
    engine = ShapExplainabilityEngine(model)
    explanation = engine.explain(features.tail(20))

    interactions = engine.interaction_summary(explanation, top_n=3)
    text = interpret_feature_effect("rolling_volatility_20", -0.01)

    assert len(interactions) <= 3
    assert {"feature_1", "feature_2", "interaction_strength"}.issubset(interactions.columns)
    assert "risk/regime" in text
    assert "reduced" in text

"""Regime-aware model training and dynamic model switching."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from models.research import ModelPrediction, PredictiveModel
from models.xgboost_factor import XGBoostFactorModel, XGBoostModelConfig

ModelFactory = Callable[[], PredictiveModel]


@dataclass(frozen=True)
class RegimeModelConfig:
    """Configuration for regime-specific predictive modeling."""

    min_regime_observations: int = 20
    fallback_regime_name: str = "fallback"


class RegimeAwareModel:
    """Train separate predictive models per regime and switch dynamically."""

    def __init__(
        self,
        model_factory: ModelFactory | None = None,
        config: RegimeModelConfig | None = None,
    ) -> None:
        self.model_factory = model_factory or _default_model_factory
        self.config = config or RegimeModelConfig()
        self.models: dict[str, PredictiveModel] = {}
        self.target_name = "target"

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        regimes: pd.Series,
    ) -> RegimeAwareModel:
        """Train one model for each sufficiently populated regime."""

        x, y, r = _align_inputs(features, target, regimes)
        self.target_name = y.name or self.target_name

        fallback_model = self.model_factory()
        fallback_model.fit(x, y)
        self.models[self.config.fallback_regime_name] = fallback_model

        for regime_name, regime_index in r.groupby(r).groups.items():
            regime_features = x.loc[regime_index]
            regime_target = y.loc[regime_index]
            if len(regime_features) < self.config.min_regime_observations:
                continue

            model = self.model_factory()
            model.fit(regime_features, regime_target)
            self.models[str(regime_name)] = model

        return self

    def predict(self, features: pd.DataFrame, regimes: pd.Series) -> ModelPrediction:
        """Predict with the model associated with each timestamp's regime."""

        aligned = features.join(regimes.rename("regime"), how="inner").dropna()
        if aligned.empty:
            raise ValueError("no aligned feature/regime rows available")

        predictions = pd.Series(index=aligned.index, dtype=float, name="regime_prediction")
        for regime_name, regime_rows in aligned.groupby("regime"):
            model = self.models.get(str(regime_name), self.models[self.config.fallback_regime_name])
            regime_features = regime_rows.drop(columns=["regime"])
            predictions.loc[regime_features.index] = model.predict(regime_features).predictions

        return ModelPrediction(
            predictions=predictions.sort_index(),
            target_name=self.target_name,
            model_name="regime_aware_model",
        )


def _default_model_factory() -> PredictiveModel:
    """Default lightweight XGBoost model used for regime-specific forecasting."""

    return XGBoostFactorModel(
        XGBoostModelConfig(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.05,
            n_jobs=1,
        )
    )


def _align_inputs(
    features: pd.DataFrame,
    target: pd.Series,
    regimes: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Align features, targets, and regimes by timestamp."""

    target_name = target.name or "target"
    aligned = (
        features.join(target.rename(target_name), how="inner")
        .join(regimes.rename("regime"), how="inner")
        .dropna()
    )
    if aligned.empty:
        raise ValueError("no aligned feature/target/regime rows available")

    return aligned.drop(columns=[target_name, "regime"]), aligned[target_name], aligned["regime"]

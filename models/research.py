"""Model interfaces for predictive return and direction forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class ModelPrediction:
    """Timestamp-aligned model output ready for signal evaluation."""

    predictions: pd.Series
    target_name: str
    model_name: str


class PredictiveModel(Protocol):
    """Protocol implemented by XGBoost, LSTM, ensemble, and regime models."""

    def fit(self, features: pd.DataFrame, target: pd.Series) -> PredictiveModel:
        """Train the model on historical, time-ordered observations."""

    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Generate timestamp-aligned predictions for future evaluation."""

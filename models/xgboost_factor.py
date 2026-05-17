"""XGBoost factor models for return and direction forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from models.research import ModelPrediction

ObjectiveType = Literal["regression", "classification"]


@dataclass(frozen=True)
class XGBoostModelConfig:
    """Configuration for an XGBoost factor model."""

    objective_type: ObjectiveType = "regression"
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    n_jobs: int = 1
    eval_metric: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)


class XGBoostFactorModel:
    """Thin production-style wrapper around XGBoost for factor research.

    The model accepts a timestamp-indexed feature matrix and returns timestamp-aligned
    predictions. Regression predicts future return magnitude. Classification predicts
    the probability of a positive future return.
    """

    def __init__(self, config: XGBoostModelConfig | None = None) -> None:
        self.config = config or XGBoostModelConfig()
        self.estimator: Any | None = None
        self.feature_names: list[str] = []
        self.target_name: str = "target"

    def fit(self, features: pd.DataFrame, target: pd.Series) -> XGBoostFactorModel:
        """Fit the XGBoost factor model on historical observations."""

        self._validate_inputs(features, target)
        self.feature_names = list(features.columns)
        self.target_name = target.name or self.target_name
        self.estimator = self._build_estimator()
        self.estimator.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """Generate timestamp-aligned model predictions."""

        if self.estimator is None:
            raise ValueError("model must be fit before calling predict")

        aligned_features = features.loc[:, self.feature_names]
        if (
            self.config.objective_type == "classification"
            and hasattr(self.estimator, "predict_proba")
        ):
            raw_predictions = self.estimator.predict_proba(aligned_features)[:, 1]
        else:
            raw_predictions = self.estimator.predict(aligned_features)

        predictions = pd.Series(
            raw_predictions,
            index=aligned_features.index,
            name=f"{self.config.objective_type}_prediction",
        )
        return ModelPrediction(
            predictions=predictions,
            target_name=self.target_name,
            model_name="xgboost_factor",
        )

    @property
    def feature_importance(self) -> pd.Series:
        """Return gain-style feature importances exposed by the fitted estimator."""

        if self.estimator is None:
            raise ValueError("model must be fit before feature importance is available")
        if not hasattr(self.estimator, "feature_importances_"):
            return pd.Series(dtype=float, name="feature_importance")

        return pd.Series(
            self.estimator.feature_importances_,
            index=self.feature_names,
            name="feature_importance",
        ).sort_values(ascending=False)

    def _build_estimator(self) -> Any:
        """Construct the concrete XGBoost estimator lazily."""

        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "Install xgboost to train XGBoostFactorModel: "
                "python -m pip install xgboost"
            ) from exc

        common_params = {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "random_state": self.config.random_state,
            "n_jobs": self.config.n_jobs,
            **self.config.extra_params,
        }

        if self.config.objective_type == "classification":
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric=self.config.eval_metric or "logloss",
                **common_params,
            )

        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric=self.config.eval_metric or "rmse",
            **common_params,
        )

    def _validate_inputs(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Validate model training inputs."""

        if features.empty:
            raise ValueError("features cannot be empty")
        if target.empty:
            raise ValueError("target cannot be empty")
        if not features.index.equals(target.index):
            raise ValueError("features and target must have identical indexes")
        if self.config.objective_type not in {"regression", "classification"}:
            raise ValueError("objective_type must be 'regression' or 'classification'")

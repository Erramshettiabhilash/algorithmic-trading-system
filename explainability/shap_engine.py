"""SHAP explainability engine for financial machine learning models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ShapExplanation:
    """Tabular SHAP output aligned to the explained feature matrix."""

    values: pd.DataFrame
    base_value: float
    data: pd.DataFrame
    model_output: str = "raw"


@dataclass(frozen=True)
class LocalShapExplanation:
    """Single-row explanation that resembles the data behind a SHAP force plot."""

    timestamp: pd.Timestamp
    base_value: float
    prediction_value: float
    contributions: pd.Series
    feature_values: pd.Series


class ShapExplainabilityEngine:
    """Compute and summarize SHAP explanations for fitted factor models."""

    def __init__(self, model: Any, *, model_output: str = "raw") -> None:
        self.model = model
        self.model_output = model_output

    def explain(self, features: pd.DataFrame) -> ShapExplanation:
        """Compute SHAP values for a timestamp-indexed feature matrix."""

        estimator = _extract_estimator(self.model)
        if features.empty:
            raise ValueError("features cannot be empty")

        shap = _import_shap()
        explainer = shap.TreeExplainer(estimator, model_output=self.model_output)
        raw_values = explainer.shap_values(features)
        values = _coerce_shap_values(raw_values, features.index, features.columns)
        base_value = _coerce_base_value(explainer.expected_value)

        return ShapExplanation(
            values=values,
            base_value=base_value,
            data=features.copy(),
            model_output=self.model_output,
        )

    def feature_importance(self, explanation: ShapExplanation) -> pd.Series:
        """Rank features by mean absolute SHAP value."""

        importance = explanation.values.abs().mean(axis=0)
        importance.name = "mean_abs_shap"
        return importance.sort_values(ascending=False)

    def dependence_data(self, explanation: ShapExplanation, feature: str) -> pd.DataFrame:
        """Return feature value and SHAP contribution pairs for dependence plots."""

        if feature not in explanation.values.columns:
            raise ValueError(f"feature '{feature}' not found in SHAP explanation")

        return pd.DataFrame(
            {
                "feature_value": explanation.data[feature],
                "shap_value": explanation.values[feature],
            },
            index=explanation.data.index,
        )

    def local_explanation(
        self,
        explanation: ShapExplanation,
        *,
        row: int | pd.Timestamp = -1,
        top_n: int = 8,
    ) -> LocalShapExplanation:
        """Return the largest positive/negative contributions for one prediction."""

        if isinstance(row, pd.Timestamp):
            contributions = explanation.values.loc[row]
            feature_values = explanation.data.loc[row]
            timestamp = row
        else:
            contributions = explanation.values.iloc[row]
            feature_values = explanation.data.iloc[row]
            timestamp = explanation.values.index[row]

        ordered = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
        top_contributions = ordered.head(top_n)
        prediction_value = float(explanation.base_value + contributions.sum())

        return LocalShapExplanation(
            timestamp=pd.Timestamp(timestamp),
            base_value=explanation.base_value,
            prediction_value=prediction_value,
            contributions=top_contributions,
            feature_values=feature_values.loc[top_contributions.index],
        )

    def interaction_summary(self, explanation: ShapExplanation, top_n: int = 10) -> pd.DataFrame:
        """Approximate feature interaction strength using SHAP contribution correlations."""

        values = explanation.values
        rows: list[dict[str, float | str]] = []
        for i, first in enumerate(values.columns):
            for second in values.columns[i + 1 :]:
                strength = abs(values[first].corr(values[second]))
                if not np.isnan(strength):
                    rows.append(
                        {
                            "feature_1": first,
                            "feature_2": second,
                            "interaction_strength": float(strength),
                        }
                    )

        if not rows:
            return pd.DataFrame(columns=["feature_1", "feature_2", "interaction_strength"])

        return (
            pd.DataFrame(rows)
            .sort_values("interaction_strength", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def save_summary_plot(
        self,
        explanation: ShapExplanation,
        output_path: str | Path,
        *,
        max_display: int = 20,
    ) -> Path:
        """Save a SHAP beeswarm-style summary plot to disk."""

        shap = _import_shap()
        import matplotlib.pyplot as plt

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        shap.summary_plot(
            explanation.values.to_numpy(),
            explanation.data,
            feature_names=list(explanation.data.columns),
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output, dpi=150, bbox_inches="tight")
        plt.close()
        return output


def interpret_feature_effect(feature: str, shap_value: float) -> str:
    """Create a concise finance-readable interpretation for one SHAP contribution."""

    direction = "increased" if shap_value > 0 else "reduced"
    family = _feature_family(feature)
    return (
        f"{feature} {direction} the model forecast. "
        f"In finance terms, this is a {family} effect with contribution {shap_value:.6f}."
    )


def _feature_family(feature: str) -> str:
    """Map a feature name to a broad factor family."""

    if "volatility" in feature or "atr" in feature:
        return "risk/regime"
    if "momentum" in feature or "macd" in feature or "ema" in feature or "rsi" in feature:
        return "momentum"
    if "volume" in feature or "obv" in feature:
        return "participation/flow"
    if "fractal" in feature or "sweep" in feature or "trend_structure" in feature:
        return "market-structure"
    if "return" in feature:
        return "return"
    return "cross-factor"


def _extract_estimator(model: Any) -> Any:
    """Extract the underlying estimator from a wrapper or accept a raw model."""

    estimator = getattr(model, "estimator", model)
    if estimator is None:
        raise ValueError("model must be fitted before SHAP explanation")
    return estimator


def _import_shap() -> Any:
    """Import SHAP lazily with a clear install hint."""

    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install shap to use explainability: python -m pip install shap") from exc
    return shap


def _coerce_shap_values(
    raw_values: Any,
    index: pd.DatetimeIndex,
    columns: pd.Index,
) -> pd.DataFrame:
    """Coerce SHAP outputs from regressors/classifiers into a DataFrame."""

    if isinstance(raw_values, list):
        raw_values = raw_values[-1]

    values = np.asarray(raw_values)
    if values.ndim == 3:
        values = values[:, :, -1]

    return pd.DataFrame(values, index=index, columns=columns)


def _coerce_base_value(expected_value: Any) -> float:
    """Convert SHAP expected value variants into one scalar."""

    values = np.asarray(expected_value).reshape(-1)
    return float(values[-1])

"""Walk-forward research pipeline and stability diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from evaluation.metrics import information_coefficient, information_ratio
from evaluation.model_report import FactorModelEvaluation, evaluate_factor_predictions
from evaluation.validation import (
    TemporalSplit,
    expanding_window_splits,
    rolling_window_splits,
)
from models.research import PredictiveModel

WindowMode = Literal["expanding", "rolling"]
ModelFactory = Callable[[], PredictiveModel]


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for walk-forward research."""

    mode: WindowMode = "expanding"
    initial_train_size: int = 252
    train_size: int = 252
    test_size: int = 21
    step_size: int | None = None
    target_type: str = "regression"


@dataclass(frozen=True)
class WalkForwardFoldResult:
    """Out-of-sample result for one walk-forward fold."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    predictions: pd.Series
    realized: pd.Series
    evaluation: FactorModelEvaluation


@dataclass(frozen=True)
class WalkForwardResult:
    """Complete walk-forward experiment output."""

    predictions: pd.Series
    realized: pd.Series
    fold_results: tuple[WalkForwardFoldResult, ...]
    evaluation: FactorModelEvaluation
    fold_summary: pd.DataFrame


def run_walk_forward_research(
    features: pd.DataFrame,
    target: pd.Series,
    model_factory: ModelFactory,
    config: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    """Run rolling or expanding retraining and collect out-of-sample predictions."""

    cfg = config or WalkForwardConfig()
    splits = list(_iter_splits(features, target, cfg))
    if not splits:
        raise ValueError("walk-forward configuration produced no validation folds")

    fold_results: list[WalkForwardFoldResult] = []
    prediction_blocks: list[pd.Series] = []
    realized_blocks: list[pd.Series] = []

    for fold, split in enumerate(splits):
        model = model_factory()
        model.fit(split.x_train, split.y_train)
        predictions = model.predict(split.x_test).predictions
        realized = split.y_test.loc[predictions.index]
        evaluation = evaluate_factor_predictions(
            predictions,
            realized,
            target_type=cfg.target_type,
        )

        fold_result = WalkForwardFoldResult(
            fold=fold,
            train_start=pd.Timestamp(split.x_train.index.min()),
            train_end=pd.Timestamp(split.x_train.index.max()),
            test_start=pd.Timestamp(predictions.index.min()),
            test_end=pd.Timestamp(predictions.index.max()),
            predictions=predictions,
            realized=realized,
            evaluation=evaluation,
        )
        fold_results.append(fold_result)
        prediction_blocks.append(predictions)
        realized_blocks.append(realized)

    predictions = pd.concat(prediction_blocks).sort_index()
    realized = pd.concat(realized_blocks).sort_index()
    evaluation = evaluate_factor_predictions(
        predictions,
        realized,
        target_type=cfg.target_type,
    )
    return WalkForwardResult(
        predictions=predictions,
        realized=realized,
        fold_results=tuple(fold_results),
        evaluation=evaluation,
        fold_summary=summarize_walk_forward_folds(fold_results),
    )


def summarize_walk_forward_folds(
    fold_results: Iterable[WalkForwardFoldResult],
) -> pd.DataFrame:
    """Create a per-fold metric table for research review."""

    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for result in fold_results:
        rows.append(
            {
                "fold": result.fold,
                "train_start": result.train_start,
                "train_end": result.train_end,
                "test_start": result.test_start,
                "test_end": result.test_end,
                "rmse": result.evaluation.rmse,
                "accuracy": result.evaluation.accuracy,
                "sharpe": result.evaluation.sharpe,
                "max_drawdown": result.evaluation.max_drawdown,
                "information_coefficient": result.evaluation.information_coefficient,
                "observations": result.evaluation.observations,
            }
        )

    return pd.DataFrame(rows).set_index("fold")


def analyze_alpha_decay(
    predictions: pd.Series,
    forward_returns_by_horizon: dict[int, pd.Series],
) -> pd.DataFrame:
    """Measure prediction IC across multiple forward-return horizons."""

    rows: list[dict[str, float | int]] = []
    for horizon, returns in sorted(forward_returns_by_horizon.items()):
        rows.append(
            {
                "horizon": horizon,
                "information_coefficient": information_coefficient(predictions, returns),
            }
        )

    frame = pd.DataFrame(rows).set_index("horizon")
    frame["abs_ic"] = frame["information_coefficient"].abs()
    frame["alpha_decay"] = frame["abs_ic"].iloc[0] - frame["abs_ic"]
    return frame


def analyze_prediction_drift(
    predictions: pd.Series,
    *,
    window: int,
) -> pd.DataFrame:
    """Track rolling prediction mean, volatility, and z-score drift."""

    if window < 2:
        raise ValueError("window must be at least 2")

    clean = predictions.dropna().sort_index()
    rolling_mean = clean.rolling(window).mean()
    rolling_std = clean.rolling(window).std(ddof=0)
    expanding_mean = clean.expanding(min_periods=window).mean()
    expanding_std = clean.expanding(min_periods=window).std(ddof=0)
    drift_zscore = (rolling_mean - expanding_mean) / expanding_std.replace(0.0, np.nan)

    return pd.DataFrame(
        {
            "prediction_mean": rolling_mean,
            "prediction_volatility": rolling_std,
            "prediction_drift_zscore": drift_zscore,
        }
    ).fillna(0.0)


def analyze_model_stability(fold_summary: pd.DataFrame) -> pd.Series:
    """Summarize fold-to-fold stability of walk-forward performance."""

    if fold_summary.empty:
        raise ValueError("fold_summary cannot be empty")

    ic_series = fold_summary["information_coefficient"]
    sharpe_series = fold_summary["sharpe"]
    return pd.Series(
        {
            "mean_ic": ic_series.mean(),
            "ic_volatility": ic_series.std(ddof=1),
            "ic_information_ratio": information_ratio(ic_series),
            "positive_ic_rate": (ic_series > 0.0).mean(),
            "mean_sharpe": sharpe_series.mean(),
            "sharpe_volatility": sharpe_series.std(ddof=1),
            "positive_sharpe_rate": (sharpe_series > 0.0).mean(),
        },
        name="model_stability",
    ).fillna(0.0)


def _iter_splits(
    features: pd.DataFrame,
    target: pd.Series,
    config: WalkForwardConfig,
) -> Iterable[TemporalSplit]:
    """Dispatch to expanding or rolling validation splits."""

    if config.mode == "expanding":
        return expanding_window_splits(
            features,
            target,
            initial_train_size=config.initial_train_size,
            test_size=config.test_size,
            step_size=config.step_size,
        )
    if config.mode == "rolling":
        return rolling_window_splits(
            features,
            target,
            train_size=config.train_size,
            test_size=config.test_size,
            step_size=config.step_size,
        )
    raise ValueError("mode must be either 'expanding' or 'rolling'")

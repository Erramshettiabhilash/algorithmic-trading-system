"""Optuna-based Bayesian optimization for quant models and trading systems."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from evaluation.model_report import FactorModelEvaluation, evaluate_factor_predictions
from evaluation.validation import temporal_train_test_split
from models.lstm import LSTMForecaster, LSTMModelConfig
from models.xgboost_factor import XGBoostFactorModel, XGBoostModelConfig
from optimization.search_space import LSTMSearchSpace, RLSearchSpace, XGBoostSearchSpace

OptimizationMetric = Literal["rmse", "ic", "sharpe"]
RLMetricEvaluator = Callable[[dict[str, Any]], float]


@dataclass(frozen=True)
class OptimizationResult:
    """Summary of a completed hyperparameter optimization study."""

    best_params: dict[str, Any]
    best_value: float
    metric: str
    n_trials: int
    direction: str


def optimize_xgboost_with_optuna(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_trials: int = 20,
    test_size: float | int = 0.2,
    objective_type: Literal["regression", "classification"] = "regression",
    metric: OptimizationMetric = "ic",
    search_space: XGBoostSearchSpace | None = None,
    random_state: int = 42,
) -> OptimizationResult:
    """Tune XGBoost parameters using a chronological validation split."""

    optuna = _import_optuna()
    space = search_space or XGBoostSearchSpace()
    split = temporal_train_test_split(features, target, test_size=test_size)
    direction = _metric_direction(metric)

    def objective(trial: Any) -> float:
        config = XGBoostModelConfig(
            objective_type=objective_type,
            n_estimators=trial.suggest_int("n_estimators", *space.n_estimators),
            max_depth=trial.suggest_int("max_depth", *space.max_depth),
            learning_rate=trial.suggest_float("learning_rate", *space.learning_rate, log=True),
            subsample=trial.suggest_float("subsample", *space.subsample),
            colsample_bytree=trial.suggest_float("colsample_bytree", *space.colsample_bytree),
            random_state=random_state,
            n_jobs=1,
        )
        model = XGBoostFactorModel(config).fit(split.x_train, split.y_train)
        predictions = model.predict(split.x_test).predictions
        evaluation = evaluate_factor_predictions(
            predictions,
            split.y_test,
            target_type=objective_type,
        )
        return _score_evaluation(evaluation, metric)

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return _result_from_study(study, metric, direction)


def optimize_lstm_with_optuna(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_trials: int = 10,
    test_size: float | int = 0.2,
    objective_type: Literal["regression", "classification"] = "regression",
    metric: OptimizationMetric = "rmse",
    search_space: LSTMSearchSpace | None = None,
    random_state: int = 42,
) -> OptimizationResult:
    """Tune an LSTM architecture using temporal validation."""

    optuna = _import_optuna()
    space = search_space or LSTMSearchSpace()
    split = temporal_train_test_split(features, target, test_size=test_size)
    direction = _metric_direction(metric)

    def objective(trial: Any) -> float:
        num_layers = trial.suggest_int("num_layers", *space.num_layers)
        dropout = trial.suggest_float("dropout", *space.dropout) if num_layers > 1 else 0.0
        config = LSTMModelConfig(
            sequence_length=trial.suggest_int("sequence_length", *space.sequence_length),
            hidden_size=trial.suggest_int("hidden_size", *space.hidden_size),
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=trial.suggest_float("learning_rate", *space.learning_rate, log=True),
            batch_size=trial.suggest_categorical("batch_size", list(space.batch_size_choices)),
            epochs=space.epochs,
            objective_type=objective_type,
            random_state=random_state,
        )
        model = LSTMForecaster(config).fit(split.x_train, split.y_train)
        predictions = model.predict(split.x_test).predictions
        aligned_target = split.y_test.loc[predictions.index]
        evaluation = evaluate_factor_predictions(
            predictions,
            aligned_target,
            target_type=objective_type,
        )
        return _score_evaluation(evaluation, metric)

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return _result_from_study(study, metric, direction)


def optimize_rl_with_optuna(
    evaluator: RLMetricEvaluator,
    *,
    n_trials: int = 20,
    search_space: RLSearchSpace | None = None,
    random_state: int = 42,
) -> OptimizationResult:
    """Optimize RL hyperparameters through a user-supplied evaluation callback.

    The callback should train or simulate an RL policy and return a scalar objective,
    usually validation Sharpe or total return. Keeping the evaluator injectable makes
    the optimizer useful for DQN, PPO, and Actor-Critic without coupling it to one
    expensive training loop.
    """

    optuna = _import_optuna()
    space = search_space or RLSearchSpace()

    def objective(trial: Any) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", *space.learning_rate, log=True),
            "gamma": trial.suggest_float("gamma", *space.gamma),
            "transaction_cost_bps": trial.suggest_float(
                "transaction_cost_bps",
                *space.transaction_cost_bps,
            ),
            "reward_window": trial.suggest_int("reward_window", *space.reward_window),
            "lookback_window": trial.suggest_int("lookback_window", *space.lookback_window),
        }
        return float(evaluator(params))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return _result_from_study(study, "rl_objective", "maximize")


def _score_evaluation(evaluation: FactorModelEvaluation, metric: OptimizationMetric) -> float:
    """Convert an evaluation object into the scalar Optuna objective."""

    if metric == "rmse":
        return evaluation.rmse
    if metric == "ic":
        return evaluation.information_coefficient
    if metric == "sharpe":
        return evaluation.sharpe
    raise ValueError("metric must be 'rmse', 'ic', or 'sharpe'")


def _metric_direction(metric: OptimizationMetric) -> str:
    """Return Optuna direction for a supported metric."""

    return "minimize" if metric == "rmse" else "maximize"


def _result_from_study(study: Any, metric: str, direction: str) -> OptimizationResult:
    """Convert an Optuna study into a stable project result object."""

    return OptimizationResult(
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        metric=metric,
        n_trials=len(study.trials),
        direction=direction,
    )


def _import_optuna() -> Any:
    """Import Optuna lazily with a clear setup message."""

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Install optuna to run Bayesian optimization.") from exc
    return optuna

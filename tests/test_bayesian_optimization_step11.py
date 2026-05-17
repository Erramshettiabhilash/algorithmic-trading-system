"""Tests for Step 11 Bayesian optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from optimization import (
    LSTMSearchSpace,
    RLSearchSpace,
    XGBoostSearchSpace,
    lstm_hyperopt_space,
    optimize_lstm_with_optuna,
    optimize_rl_with_optuna,
    optimize_xgboost_with_optuna,
    rl_hyperopt_space,
    xgboost_hyperopt_space,
)


def _factor_dataset(rows: int = 80) -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B", tz="UTC")
    momentum = np.linspace(-1.0, 1.0, rows)
    cycle = np.sin(np.linspace(0.0, 8.0, rows))
    volatility = np.cos(np.linspace(0.0, 5.0, rows))
    target = 0.04 * momentum + 0.02 * cycle - 0.01 * volatility

    features = pd.DataFrame(
        {
            "momentum_factor_10": momentum,
            "macd_histogram": cycle,
            "rolling_volatility_20": volatility,
        },
        index=dates,
    )
    return features, pd.Series(target, index=dates, name="forward_return_1")


def test_optuna_xgboost_optimizer_returns_best_params() -> None:
    features, target = _factor_dataset()
    result = optimize_xgboost_with_optuna(
        features,
        target,
        n_trials=2,
        test_size=15,
        metric="ic",
        search_space=XGBoostSearchSpace(
            n_estimators=(5, 10),
            max_depth=(1, 2),
            learning_rate=(0.05, 0.20),
            subsample=(0.8, 1.0),
            colsample_bytree=(0.8, 1.0),
        ),
    )

    assert result.n_trials == 2
    assert result.direction == "maximize"
    assert "max_depth" in result.best_params


def test_optuna_lstm_optimizer_returns_best_params() -> None:
    features, target = _factor_dataset(rows=70)
    result = optimize_lstm_with_optuna(
        features,
        target,
        n_trials=2,
        test_size=20,
        metric="rmse",
        search_space=LSTMSearchSpace(
            sequence_length=(4, 6),
            hidden_size=(4, 8),
            num_layers=(1, 1),
            dropout=(0.0, 0.0),
            learning_rate=(0.005, 0.02),
            batch_size_choices=(8,),
            epochs=2,
        ),
    )

    assert result.n_trials == 2
    assert result.direction == "minimize"
    assert "sequence_length" in result.best_params


def test_optuna_rl_optimizer_uses_evaluator_callback() -> None:
    def evaluator(params: dict[str, float]) -> float:
        learning_rate_penalty = abs(params["learning_rate"] - 0.001)
        gamma_bonus = params["gamma"]
        return float(gamma_bonus - learning_rate_penalty * 100.0)

    result = optimize_rl_with_optuna(
        evaluator,
        n_trials=3,
        search_space=RLSearchSpace(
            learning_rate=(0.0005, 0.002),
            gamma=(0.95, 0.99),
            transaction_cost_bps=(0.0, 2.0),
            reward_window=(5, 8),
            lookback_window=(5, 8),
        ),
    )

    assert result.n_trials == 3
    assert result.direction == "maximize"
    assert "gamma" in result.best_params


def test_hyperopt_search_spaces_are_constructible() -> None:
    xgb_space = xgboost_hyperopt_space()
    lstm_space = lstm_hyperopt_space()
    rl_space = rl_hyperopt_space()

    assert {"n_estimators", "learning_rate"}.issubset(xgb_space)
    assert {"sequence_length", "hidden_size"}.issubset(lstm_space)
    assert {"gamma", "reward_window"}.issubset(rl_space)

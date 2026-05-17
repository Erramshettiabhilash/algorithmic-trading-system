"""Tests for Step 8 reinforcement-learning trading environment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from rl import (
    TradingAction,
    TradingEnvironment,
    TradingEnvironmentConfig,
    compare_rl_to_signal_strategy,
    create_rl_agent,
    evaluate_strategy_returns,
    rollout_policy,
)


def _prices(rows: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B", tz="UTC")
    close = 100.0 + np.cumsum(0.2 + 0.5 * np.sin(np.linspace(0.0, 8.0, rows)))
    open_ = close - 0.1
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = 1_000_000 + 10_000 * np.cos(np.linspace(0.0, 6.0, rows))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


class _AlwaysBuyPolicy:
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        del observation, deterministic
        return int(TradingAction.BUY), None


def test_trading_environment_is_gym_compatible() -> None:
    env = TradingEnvironment(
        _prices(),
        TradingEnvironmentConfig(lookback_window=5, reward_window=5),
    )

    check_env(env, skip_render_check=True)


def test_environment_step_updates_position_equity_and_returns() -> None:
    env = TradingEnvironment(
        _prices(),
        TradingEnvironmentConfig(lookback_window=5, reward_window=5, transaction_cost_bps=2.0),
    )
    observation, info = env.reset()
    next_observation, reward, terminated, truncated, next_info = env.step(int(TradingAction.BUY))

    assert observation.shape == env.observation_space.shape
    assert next_observation.shape == env.observation_space.shape
    assert next_info["position"] == 1.0
    assert len(env.strategy_returns()) == 1
    assert reward != 0.0
    assert not terminated
    assert not truncated
    assert info["equity"] == env.config.initial_cash


def test_sell_action_respects_no_short_configuration() -> None:
    env = TradingEnvironment(
        _prices(),
        TradingEnvironmentConfig(lookback_window=5, allow_short=False),
    )
    env.reset()
    env.step(int(TradingAction.SELL))

    assert env.position == 0.0


def test_strategy_evaluation_and_policy_rollout() -> None:
    env = TradingEnvironment(
        _prices(),
        TradingEnvironmentConfig(lookback_window=5, reward_window=5),
    )
    returns = rollout_policy(env, _AlwaysBuyPolicy())
    evaluation = evaluate_strategy_returns(returns)

    assert len(returns) > 0
    assert evaluation.observations == len(returns)
    assert evaluation.max_drawdown <= 0.0


def test_compare_rl_to_signal_strategy_returns_metric_table() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    rl_returns = pd.Series([0.01, -0.002, 0.003, 0.004, -0.001] * 2, index=index)
    signal_returns = pd.Series([0.004, 0.001, -0.002, 0.002, 0.003] * 2, index=index)

    comparison = compare_rl_to_signal_strategy(rl_returns, signal_returns)

    assert {"rl_strategy", "signal_strategy"}.issubset(comparison.index)
    assert "sharpe" in comparison.columns


def test_rl_agent_factories_create_supported_algorithms() -> None:
    env = TradingEnvironment(
        _prices(),
        TradingEnvironmentConfig(lookback_window=5, reward_window=5),
    )

    dqn = create_rl_agent(
        "dqn",
        env,
        learning_starts=5,
        buffer_size=100,
        batch_size=8,
        train_freq=1,
        gradient_steps=1,
    )
    ppo = create_rl_agent("ppo", env, n_steps=8, batch_size=4)
    actor_critic = create_rl_agent("actor_critic", env, n_steps=8)

    assert dqn.__class__.__name__ == "DQN"
    assert ppo.__class__.__name__ == "PPO"
    assert actor_critic.__class__.__name__ == "A2C"

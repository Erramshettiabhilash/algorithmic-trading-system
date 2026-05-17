# Step 8 - Reinforcement Learning Trading

## Goal

Build a reinforcement learning trading layer where an agent learns sequential Buy/Sell/Hold
decisions from portfolio feedback.

## Why RL Is Different From Forecasting

XGBoost and LSTM predict what may happen next. RL decides what to do next.

In trading, this matters because a decision depends on the current position, transaction costs,
recent volatility, and the sequence of previous actions. A good forecast is not automatically a
good trading policy.

## Implemented Components

- `TradingEnvironment`: Gymnasium-compatible single-asset trading environment.
- `TradingAction`: discrete action space: Sell, Hold, Buy.
- `TradingEnvironmentConfig`: cash, costs, lookback, reward, position sizing.
- `create_rl_agent`: factory for DQN, PPO, and Actor-Critic agents.
- `train_rl_agent`: small training wrapper.
- `rollout_policy`: evaluates a trained policy through the environment.
- `evaluate_strategy_returns`: Sharpe, drawdown, total return.
- `compare_rl_to_signal_strategy`: compares RL policy returns to signal-model returns.

## State Space

The state is a flattened rolling window of normalized OHLCV values plus the current position.

```text
normalized OHLCV[t-lookback : t] + current_position
```

This gives the agent market context and portfolio context.

## Action Space

```text
Sell = 0
Hold = 1
Buy  = 2
```

Sell targets short exposure when shorting is allowed. Hold preserves the current position. Buy
targets long exposure.

## Reward Engineering

The environment charges transaction costs when exposure changes:

```text
portfolio_return = position * market_return - transaction_cost
```

The reward is Sharpe-like after enough observations:

```text
rolling_mean_return / rolling_return_volatility
```

This matters because an RL agent should not be rewarded only for raw return. It should learn that
unstable returns and overtrading are costly.

## Agents

DQN is a value-based algorithm for discrete actions. It learns the expected value of Buy, Sell, or
Hold.

PPO is a policy-gradient algorithm that is often more stable than older policy methods.

Actor-Critic uses one model component to choose actions and another to estimate value. In
Stable-Baselines3 this is represented here with A2C.

## Minimal Example

```python
from rl import TradingEnvironment, TradingEnvironmentConfig, create_rl_agent, train_rl_agent

env = TradingEnvironment(
    prices,
    TradingEnvironmentConfig(
        lookback_window=20,
        transaction_cost_bps=1.0,
        reward_window=20,
    ),
)

agent = create_rl_agent("ppo", env)
train_rl_agent(agent, total_timesteps=10_000)
```

## RL vs XGBoost Strategy Performance

To compare RL against a forecast model, evaluate both as return streams:

```python
from rl import compare_rl_to_signal_strategy, evaluate_strategy_returns, rollout_policy

rl_returns = rollout_policy(env, agent)
comparison = compare_rl_to_signal_strategy(rl_returns, xgboost_signal_returns)
print(comparison)
```

This keeps the comparison honest: both systems are judged by trading outcomes.

## Exploration vs Exploitation

Exploration means trying actions that may not currently look optimal. Exploitation means choosing
the action the agent currently believes is best. Trading RL struggles because exploration can be
expensive: bad actions lose money.

In research, exploration happens in simulation only. Live systems should use heavily validated,
constrained policies.

## Why RL Trading Is Unstable

RL trading is difficult because:

- market regimes change
- rewards are noisy
- transaction costs punish overtrading
- small environment assumptions can dominate results
- agents can exploit backtest artifacts
- training can be seed-sensitive

For this reason, RL should be treated as an advanced research layer, not a shortcut to profitable
trading.

## Interview-Ready Explanation

I built a custom Gymnasium trading environment where the state is a rolling OHLCV window plus
current position, actions are Buy/Sell/Hold, and rewards include transaction costs and a Sharpe-like
risk adjustment. I use Stable-Baselines3 for DQN, PPO, and Actor-Critic agents because the core RL
algorithms are well-tested. The key research question is whether an RL policy improves realized
trading metrics over an ML signal strategy, not whether it simply learns a high simulation reward.

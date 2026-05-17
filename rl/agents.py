"""Stable-Baselines agent factories for trading environments."""

from __future__ import annotations

from typing import Any, Literal

AgentName = Literal["dqn", "ppo", "actor_critic"]


def create_rl_agent(
    agent_name: AgentName,
    env: Any,
    *,
    learning_rate: float = 0.0003,
    seed: int = 42,
    verbose: int = 0,
    **kwargs: Any,
) -> Any:
    """Create a DQN, PPO, or Actor-Critic trading agent."""

    try:
        from stable_baselines3 import A2C, DQN, PPO
    except ImportError as exc:
        raise ImportError(
            "Install stable-baselines3 to create RL agents: "
            "python -m pip install stable-baselines3"
        ) from exc

    common = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": learning_rate,
        "seed": seed,
        "verbose": verbose,
        **kwargs,
    }

    if agent_name == "dqn":
        return DQN(**common)
    if agent_name == "ppo":
        return PPO(**common)
    if agent_name == "actor_critic":
        return A2C(**common)

    raise ValueError("agent_name must be 'dqn', 'ppo', or 'actor_critic'")


def train_rl_agent(agent: Any, *, total_timesteps: int) -> Any:
    """Train a Stable-Baselines agent and return it."""

    if total_timesteps < 1:
        raise ValueError("total_timesteps must be positive")
    return agent.learn(total_timesteps=total_timesteps)

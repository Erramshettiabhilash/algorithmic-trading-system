"""Reusable hyperparameter search-space definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class XGBoostSearchSpace:
    """Conservative starter search space for tree-based factor models."""

    n_estimators: tuple[int, int] = (100, 800)
    max_depth: tuple[int, int] = (2, 8)
    learning_rate: tuple[float, float] = (0.005, 0.20)
    subsample: tuple[float, float] = (0.50, 1.00)
    colsample_bytree: tuple[float, float] = (0.50, 1.00)


@dataclass(frozen=True)
class LSTMSearchSpace:
    """Compact architecture search space for sequence models."""

    sequence_length: tuple[int, int] = (5, 30)
    hidden_size: tuple[int, int] = (8, 64)
    num_layers: tuple[int, int] = (1, 3)
    dropout: tuple[float, float] = (0.0, 0.30)
    learning_rate: tuple[float, float] = (0.0003, 0.03)
    batch_size_choices: tuple[int, ...] = (8, 16, 32, 64)
    epochs: int = 10


@dataclass(frozen=True)
class RLSearchSpace:
    """Search space for reinforcement-learning agent and environment parameters."""

    learning_rate: tuple[float, float] = (0.00005, 0.003)
    gamma: tuple[float, float] = (0.90, 0.999)
    transaction_cost_bps: tuple[float, float] = (0.0, 5.0)
    reward_window: tuple[int, int] = (5, 40)
    lookback_window: tuple[int, int] = (5, 40)

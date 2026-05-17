"""Hyperopt-compatible search-space helpers."""

from __future__ import annotations

import math
from typing import Any

from optimization.search_space import LSTMSearchSpace, RLSearchSpace, XGBoostSearchSpace


def xgboost_hyperopt_space(space: XGBoostSearchSpace | None = None) -> dict[str, Any]:
    """Return a Hyperopt search space for XGBoost parameters."""

    hp = _import_hyperopt_hp()
    cfg = space or XGBoostSearchSpace()
    return {
        "n_estimators": hp.quniform("n_estimators", *cfg.n_estimators, 1),
        "max_depth": hp.quniform("max_depth", *cfg.max_depth, 1),
        "learning_rate": hp.loguniform(
            "learning_rate",
            math.log(cfg.learning_rate[0]),
            math.log(cfg.learning_rate[1]),
        ),
        "subsample": hp.uniform("subsample", *cfg.subsample),
        "colsample_bytree": hp.uniform("colsample_bytree", *cfg.colsample_bytree),
    }


def lstm_hyperopt_space(space: LSTMSearchSpace | None = None) -> dict[str, Any]:
    """Return a Hyperopt search space for LSTM architecture parameters."""

    hp = _import_hyperopt_hp()
    cfg = space or LSTMSearchSpace()
    return {
        "sequence_length": hp.quniform("sequence_length", *cfg.sequence_length, 1),
        "hidden_size": hp.quniform("hidden_size", *cfg.hidden_size, 1),
        "num_layers": hp.quniform("num_layers", *cfg.num_layers, 1),
        "dropout": hp.uniform("dropout", *cfg.dropout),
        "learning_rate": hp.loguniform(
            "lstm_learning_rate",
            math.log(cfg.learning_rate[0]),
            math.log(cfg.learning_rate[1]),
        ),
        "batch_size": hp.choice("batch_size", list(cfg.batch_size_choices)),
    }


def rl_hyperopt_space(space: RLSearchSpace | None = None) -> dict[str, Any]:
    """Return a Hyperopt search space for RL policy/environment parameters."""

    hp = _import_hyperopt_hp()
    cfg = space or RLSearchSpace()
    return {
        "learning_rate": hp.loguniform(
            "rl_learning_rate",
            math.log(cfg.learning_rate[0]),
            math.log(cfg.learning_rate[1]),
        ),
        "gamma": hp.uniform("gamma", *cfg.gamma),
        "transaction_cost_bps": hp.uniform("transaction_cost_bps", *cfg.transaction_cost_bps),
        "reward_window": hp.quniform("reward_window", *cfg.reward_window, 1),
        "lookback_window": hp.quniform("lookback_window", *cfg.lookback_window, 1),
    }


def _import_hyperopt_hp() -> Any:
    """Import Hyperopt search-space primitives lazily."""

    try:
        from hyperopt import hp
    except ImportError as exc:
        raise ImportError("Install hyperopt to create Hyperopt search spaces.") from exc
    return hp

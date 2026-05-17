"""Target construction utilities for predictive financial research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetSpec:
    """Description of a forecast target used in an experiment."""

    name: str
    horizon: int
    target_type: str
    threshold: float = 0.0


def create_forward_return_target(
    prices: pd.DataFrame,
    *,
    horizon: int = 1,
    price_column: str = "close",
    log_return: bool = True,
) -> pd.Series:
    """Create a future return target aligned to the decision timestamp.

    At timestamp ``t``, the target represents the return from ``t`` to ``t + horizon``.
    Features at ``t`` can be used to predict it without looking into the future.
    """

    _validate_horizon(horizon)
    if price_column not in prices.columns:
        raise ValueError(f"price column '{price_column}' not found")

    close = prices[price_column].astype(float)
    future_close = close.shift(-horizon)
    if log_return:
        target = np.log(future_close / close)
        target.name = f"forward_log_return_{horizon}"
    else:
        target = future_close / close - 1.0
        target.name = f"forward_return_{horizon}"

    return target


def create_direction_target(
    prices: pd.DataFrame,
    *,
    horizon: int = 1,
    price_column: str = "close",
    threshold: float = 0.0,
) -> pd.Series:
    """Create a binary next-period direction target.

    The label is 1 when the future return is greater than ``threshold`` and 0 otherwise.
    A positive threshold can be used to ignore tiny moves that may be overwhelmed by costs.
    """

    forward_return = create_forward_return_target(
        prices,
        horizon=horizon,
        price_column=price_column,
        log_return=False,
    )
    direction = (forward_return > threshold).astype(float)
    direction[forward_return.isna()] = np.nan
    direction.name = f"forward_direction_{horizon}"
    return direction


def build_supervised_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align feature rows and target values into a supervised-learning dataset."""

    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features must use a DatetimeIndex")
    if not isinstance(target.index, pd.DatetimeIndex):
        raise TypeError("target must use a DatetimeIndex")

    aligned = features.join(target.rename(target.name or "target"), how="inner")
    if drop_na:
        aligned = aligned.dropna()
    if aligned.empty:
        raise ValueError("supervised dataset is empty after feature/target alignment")

    target_name = target.name or "target"
    return aligned.drop(columns=[target_name]), aligned[target_name]


def _validate_horizon(horizon: int) -> None:
    """Validate forecast horizon."""

    if horizon < 1:
        raise ValueError("horizon must be at least 1")

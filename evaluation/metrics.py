"""Core risk-adjusted metrics used throughout the research platform."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe ratio from periodic strategy returns."""

    clean = returns.dropna()
    volatility = clean.std(ddof=1)
    if clean.empty or volatility == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * clean.mean() / volatility)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Compute the most severe peak-to-trough loss from an equity curve."""

    clean = equity_curve.dropna()
    if clean.empty:
        return 0.0
    running_peak = clean.cummax()
    drawdown = clean / running_peak - 1.0
    return float(drawdown.min())


def information_coefficient(predictions: pd.Series, realized_returns: pd.Series) -> float:
    """Compute Spearman rank IC between predictions and future realized returns."""

    aligned = pd.concat([predictions, realized_returns], axis=1).dropna()
    if aligned.empty:
        return 0.0
    correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
    if pd.isna(correlation):
        return 0.0
    return float(correlation)


def rolling_information_coefficient(
    predictions: pd.Series,
    realized_returns: pd.Series,
    *,
    window: int,
) -> pd.Series:
    """Compute rolling Spearman IC through time."""

    if window < 2:
        raise ValueError("window must be at least 2")

    aligned = pd.concat([predictions, realized_returns], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float, name=f"rolling_ic_{window}")

    rolling_ic = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])
    rolling_ic.name = f"rolling_ic_{window}"
    return rolling_ic


def information_ratio(values: pd.Series, periods_per_year: int | None = None) -> float:
    """Compute information ratio from active returns or an IC time series.

    If ``periods_per_year`` is supplied, the ratio is annualized. For IC analysis,
    researchers often report the unannualized ICIR as mean IC divided by IC volatility.
    """

    clean = values.dropna()
    volatility = clean.std(ddof=1)
    if clean.empty or volatility == 0:
        return 0.0

    ratio = clean.mean() / volatility
    if periods_per_year is not None:
        ratio *= np.sqrt(periods_per_year)
    return float(ratio)

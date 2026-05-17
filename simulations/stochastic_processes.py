"""Stochastic process utilities used by pricing and simulation engines.

This module starts with the two core processes behind Black-Scholes:

- Brownian motion, the continuous-time random shock process.
- Geometric Brownian motion, the lognormal stock-price process.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from utils.validation import require_non_negative, require_positive

FloatArray = NDArray[np.float64]


def time_grid(maturity: float, steps: int) -> FloatArray:
    """Return an evenly spaced time grid from 0 to maturity.

    Args:
        maturity: Time horizon in years.
        steps: Number of time steps.

    Returns:
        Array with ``steps + 1`` points, including both 0 and maturity.
    """
    require_positive(maturity, "maturity")
    if steps <= 0:
        raise ValueError(f"steps must be positive. Received {steps}.")

    return np.linspace(0.0, maturity, steps + 1, dtype=np.float64)


def brownian_increments(
    maturity: float,
    steps: int,
    n_paths: int,
    seed: int | None = None,
) -> FloatArray:
    """Simulate Brownian increments.

    Brownian motion satisfies ``dW ~ N(0, dt)``. In discrete simulations this
    becomes ``Delta W = sqrt(dt) * Z`` where ``Z`` is standard normal.

    Args:
        maturity: Time horizon in years.
        steps: Number of time steps.
        n_paths: Number of independent simulated paths.
        seed: Optional random seed for reproducibility.

    Returns:
        Matrix of shape ``(n_paths, steps)`` containing Brownian increments.
    """
    require_positive(maturity, "maturity")
    if steps <= 0:
        raise ValueError(f"steps must be positive. Received {steps}.")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive. Received {n_paths}.")

    rng = np.random.default_rng(seed)
    dt = maturity / steps
    return rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, steps)).astype(np.float64)


def brownian_motion_paths(
    maturity: float,
    steps: int,
    n_paths: int,
    seed: int | None = None,
) -> FloatArray:
    """Simulate Brownian motion paths starting at zero.

    Args:
        maturity: Time horizon in years.
        steps: Number of time steps.
        n_paths: Number of independent simulated paths.
        seed: Optional random seed for reproducibility.

    Returns:
        Matrix of shape ``(n_paths, steps + 1)``. The first column is zero.
    """
    increments = brownian_increments(maturity=maturity, steps=steps, n_paths=n_paths, seed=seed)
    paths = np.concatenate(
        [np.zeros((n_paths, 1), dtype=np.float64), np.cumsum(increments, axis=1)],
        axis=1,
    )
    return paths


def geometric_brownian_motion_paths(
    spot: float,
    drift: float,
    volatility: float,
    maturity: float,
    steps: int,
    n_paths: int,
    seed: int | None = None,
) -> FloatArray:
    """Simulate stock paths under geometric Brownian motion.

    The continuous-time model is ``dS_t = mu S_t dt + sigma S_t dW_t``.
    The exact discrete solution is:

    ``S_t = S_0 exp((mu - 0.5 sigma^2)t + sigma W_t)``.

    Args:
        spot: Initial stock price.
        drift: Annualized drift. Under risk-neutral pricing this is usually
            the risk-free rate, adjusted for dividends when present.
        volatility: Annualized volatility as a decimal.
        maturity: Time horizon in years.
        steps: Number of time steps.
        n_paths: Number of independent simulated paths.
        seed: Optional random seed for reproducibility.

    Returns:
        Matrix of simulated stock prices with shape ``(n_paths, steps + 1)``.
    """
    require_positive(spot, "spot")
    require_non_negative(volatility, "volatility")

    t = time_grid(maturity=maturity, steps=steps)
    w_t = brownian_motion_paths(maturity=maturity, steps=steps, n_paths=n_paths, seed=seed)
    exponent = (drift - 0.5 * volatility**2) * t + volatility * w_t
    return spot * np.exp(exponent)


def log_returns(prices: FloatArray) -> FloatArray:
    """Compute log returns from a one-dimensional price series.

    Args:
        prices: Positive price series.

    Returns:
        Log returns ``log(S_t / S_{t-1})``.
    """
    if prices.ndim != 1:
        raise ValueError("prices must be a one-dimensional array.")
    if prices.size < 2:
        raise ValueError("prices must contain at least two observations.")
    if np.any(prices <= 0):
        raise ValueError("prices must be strictly positive.")

    return np.diff(np.log(prices))


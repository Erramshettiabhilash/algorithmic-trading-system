"""Monte Carlo simulation engine for European vanilla options."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np
from numpy.typing import NDArray

from simulations.stochastic_processes import FloatArray, geometric_brownian_motion_paths
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment


@dataclass(frozen=True)
class MonteCarloResult:
    """Result object for a Monte Carlo option-pricing run.

    Args:
        price: Discounted average payoff.
        standard_error: Standard error of the discounted payoff estimator.
        confidence_interval: Approximate 95% confidence interval.
        n_paths: Number of simulated paths.
        seed: Random seed used for reproducibility.
    """

    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    n_paths: int
    seed: int | None


@dataclass(frozen=True)
class MonteCarloEngine:
    """Vectorized GBM Monte Carlo engine for European option pricing."""

    option: EuropeanOption
    market: MarketEnvironment
    n_paths: int = 50_000
    steps: int = 252
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate simulation controls."""
        if self.n_paths <= 1:
            raise ValueError(f"n_paths must be greater than 1. Received {self.n_paths}.")
        if self.steps <= 0:
            raise ValueError(f"steps must be positive. Received {self.steps}.")

    @property
    def discount_factor(self) -> float:
        """Return the continuous-compounding discount factor ``exp(-rT)``."""
        return exp(-self.market.risk_free_rate * self.option.maturity)

    def simulate_paths(self) -> FloatArray:
        """Generate risk-neutral GBM paths for the underlying.

        Under risk-neutral pricing, the stock drift is the risk-free rate in
        this no-dividend implementation.
        """
        return geometric_brownian_motion_paths(
            spot=self.market.spot,
            drift=self.market.risk_free_rate,
            volatility=self.market.volatility,
            maturity=self.option.maturity,
            steps=self.steps,
            n_paths=self.n_paths,
            seed=self.seed,
        )

    def payoff(self, terminal_prices: NDArray[np.float64]) -> FloatArray:
        """Return option payoffs from terminal underlying prices."""
        if self.option.option_type is OptionType.CALL:
            return np.maximum(terminal_prices - self.option.strike, 0.0).astype(np.float64)
        if self.option.option_type is OptionType.PUT:
            return np.maximum(self.option.strike - terminal_prices, 0.0).astype(np.float64)

        raise ValueError(f"Unsupported option type: {self.option.option_type}.")

    def price(self) -> MonteCarloResult:
        """Estimate the option price from discounted simulated payoffs."""
        paths = self.simulate_paths()
        terminal_prices = paths[:, -1]
        payoffs = self.payoff(terminal_prices)
        discounted_payoffs = self.discount_factor * payoffs

        price = float(np.mean(discounted_payoffs))
        standard_error = float(np.std(discounted_payoffs, ddof=1) / sqrt(self.n_paths))
        margin = 1.96 * standard_error

        return MonteCarloResult(
            price=price,
            standard_error=standard_error,
            confidence_interval=(price - margin, price + margin),
            n_paths=self.n_paths,
            seed=self.seed,
        )


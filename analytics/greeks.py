"""Black-Scholes Greeks for European vanilla options."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi, sqrt

from models.black_scholes import BlackScholesModel, standard_normal_cdf
from utils.instruments import OptionType


def standard_normal_pdf(x: float) -> float:
    """Return the standard normal probability density function."""
    return exp(-0.5 * x**2) / sqrt(2.0 * pi)


@dataclass(frozen=True)
class Greeks:
    """Container for the main first- and second-order option sensitivities.

    Vega is quoted per 1.00 volatility move. Divide by 100 to get the
    approximate price change for a 1 volatility-point move.

    Rho is quoted per 1.00 rate move. Divide by 100 to get the approximate
    price change for a 1 percentage-point rate move.
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    @property
    def vega_per_vol_point(self) -> float:
        """Return Vega for a 1 volatility-point move, for example 20% to 21%."""
        return self.vega / 100.0

    @property
    def rho_per_rate_point(self) -> float:
        """Return Rho for a 1 percentage-point rate move, for example 5% to 6%."""
        return self.rho / 100.0


@dataclass(frozen=True)
class GreeksEngine:
    """Computes Black-Scholes Greeks for a European option model."""

    model: BlackScholesModel

    def delta(self) -> float:
        """Return Delta, the option price sensitivity to the underlying price."""
        n_d1 = standard_normal_cdf(self.model.d1)

        if self.model.option.option_type is OptionType.CALL:
            return n_d1
        if self.model.option.option_type is OptionType.PUT:
            return n_d1 - 1.0

        raise ValueError(f"Unsupported option type: {self.model.option.option_type}.")

    def gamma(self) -> float:
        """Return Gamma, the sensitivity of Delta to the underlying price."""
        denominator = (
            self.model.market.spot
            * self.model.market.volatility
            * sqrt(self.model.option.maturity)
        )
        return standard_normal_pdf(self.model.d1) / denominator

    def vega(self) -> float:
        """Return Vega, the option price sensitivity to volatility."""
        return (
            self.model.market.spot
            * standard_normal_pdf(self.model.d1)
            * sqrt(self.model.option.maturity)
        )

    def theta(self) -> float:
        """Return annual Theta, the option price sensitivity to calendar time."""
        common_decay = -(
            self.model.market.spot
            * standard_normal_pdf(self.model.d1)
            * self.model.market.volatility
        ) / (2.0 * sqrt(self.model.option.maturity))
        strike_term = (
            self.model.market.risk_free_rate
            * self.model.option.strike
            * self.model.discount_factor
        )

        if self.model.option.option_type is OptionType.CALL:
            return common_decay - strike_term * standard_normal_cdf(self.model.d2)
        if self.model.option.option_type is OptionType.PUT:
            return common_decay + strike_term * standard_normal_cdf(-self.model.d2)

        raise ValueError(f"Unsupported option type: {self.model.option.option_type}.")

    def theta_per_day(self, trading_days: int = 252) -> float:
        """Return approximate Theta per trading day."""
        if trading_days <= 0:
            raise ValueError(f"trading_days must be positive. Received {trading_days}.")

        return self.theta() / trading_days

    def rho(self) -> float:
        """Return Rho, the option price sensitivity to the risk-free rate."""
        strike_maturity = (
            self.model.option.strike
            * self.model.option.maturity
            * self.model.discount_factor
        )

        if self.model.option.option_type is OptionType.CALL:
            return strike_maturity * standard_normal_cdf(self.model.d2)
        if self.model.option.option_type is OptionType.PUT:
            return -strike_maturity * standard_normal_cdf(-self.model.d2)

        raise ValueError(f"Unsupported option type: {self.model.option.option_type}.")

    def all(self) -> Greeks:
        """Return all core Greeks in one immutable result object."""
        return Greeks(
            delta=self.delta(),
            gamma=self.gamma(),
            vega=self.vega(),
            theta=self.theta(),
            rho=self.rho(),
        )


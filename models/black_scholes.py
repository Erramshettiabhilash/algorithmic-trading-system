"""Black-Scholes pricing for European vanilla options.

The implementation is intentionally explicit. We use standard Python math
functions rather than a financial library so each piece of the formula is
visible and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt

from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment
from utils.validation import require_non_negative

SQRT_TWO = sqrt(2.0)


def standard_normal_cdf(x: float) -> float:
    """Return the standard normal cumulative distribution function.

    The standard normal CDF is:

    ``N(x) = P(Z <= x)`` where ``Z ~ N(0, 1)``.

    Args:
        x: Point at which to evaluate the CDF.

    Returns:
        Probability that a standard normal random variable is less than or
        equal to ``x``.
    """
    return 0.5 * (1.0 + erf(x / SQRT_TWO))


@dataclass(frozen=True)
class BlackScholesModel:
    """European option pricing model under lognormal stock dynamics.

    Args:
        option: European option contract.
        market: Current market inputs.

    The model assumes:

    - The underlying follows geometric Brownian motion.
    - Volatility is constant.
    - The option is European and can only be exercised at maturity.
    - The risk-free rate is continuously compounded.
    - There are no dividends in this first implementation.
    """

    option: EuropeanOption
    market: MarketEnvironment

    def __post_init__(self) -> None:
        """Validate model-level inputs."""
        require_non_negative(self.market.volatility, "volatility")

    @property
    def d1(self) -> float:
        """Return the Black-Scholes ``d1`` term.

        ``N(d1)`` is closely related to the option's delta and can be read as a
        risk-adjusted moneyness measure.
        """
        self._require_positive_volatility()
        numerator = log(self.market.spot / self.option.strike) + (
            self.market.risk_free_rate + 0.5 * self.market.volatility**2
        ) * self.option.maturity
        denominator = self.market.volatility * sqrt(self.option.maturity)
        return numerator / denominator

    @property
    def d2(self) -> float:
        """Return the Black-Scholes ``d2`` term.

        ``N(d2)`` is the risk-neutral probability-like term used with the
        discounted strike in the call formula.
        """
        return self.d1 - self.market.volatility * sqrt(self.option.maturity)

    @property
    def discount_factor(self) -> float:
        """Return the continuous-compounding discount factor ``exp(-rT)``."""
        return exp(-self.market.risk_free_rate * self.option.maturity)

    def price(self) -> float:
        """Return the Black-Scholes price for a European call or put."""
        if self.market.volatility == 0.0:
            return self._deterministic_discounted_payoff()

        if self.option.option_type is OptionType.CALL:
            return self._call_price()
        if self.option.option_type is OptionType.PUT:
            return self._put_price()

        raise ValueError(f"Unsupported option type: {self.option.option_type}.")

    def _call_price(self) -> float:
        """Return the European call price."""
        return (
            self.market.spot * standard_normal_cdf(self.d1)
            - self.option.strike * self.discount_factor * standard_normal_cdf(self.d2)
        )

    def _put_price(self) -> float:
        """Return the European put price."""
        return (
            self.option.strike * self.discount_factor * standard_normal_cdf(-self.d2)
            - self.market.spot * standard_normal_cdf(-self.d1)
        )

    def _deterministic_discounted_payoff(self) -> float:
        """Return the discounted payoff when volatility is exactly zero."""
        forward = self.market.spot * exp(self.market.risk_free_rate * self.option.maturity)

        if self.option.option_type is OptionType.CALL:
            payoff = max(forward - self.option.strike, 0.0)
        elif self.option.option_type is OptionType.PUT:
            payoff = max(self.option.strike - forward, 0.0)
        else:
            raise ValueError(f"Unsupported option type: {self.option.option_type}.")

        return self.discount_factor * payoff

    def _require_positive_volatility(self) -> None:
        """Raise if ``d1`` or ``d2`` is requested with zero volatility."""
        if self.market.volatility == 0.0:
            raise ValueError("d1 and d2 are undefined when volatility is zero.")


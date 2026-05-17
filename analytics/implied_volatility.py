"""Implied volatility solvers and smile extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

import pandas as pd

from analytics.greeks import GreeksEngine
from models.black_scholes import BlackScholesModel
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment
from utils.validation import require_non_negative, require_positive


@dataclass(frozen=True)
class ImpliedVolatilityResult:
    """Result object returned by an implied-volatility solver."""

    implied_volatility: float
    iterations: int
    converged: bool
    method: str
    pricing_error: float


@dataclass(frozen=True)
class ImpliedVolatilitySolver:
    """Solves for volatility from an observed option market price.

    Args:
        option: European option contract.
        market: Market inputs. The volatility field is used only as a default
            initial guess; candidate volatilities are inserted during solving.
        market_price: Observed option price to invert.
        tolerance: Absolute pricing-error tolerance.
        max_iterations: Maximum iterations for iterative solvers.
    """

    option: EuropeanOption
    market: MarketEnvironment
    market_price: float
    tolerance: float = 1e-8
    max_iterations: int = 100

    def __post_init__(self) -> None:
        """Validate solver inputs and no-arbitrage bounds."""
        require_non_negative(self.market_price, "market_price")
        require_positive(self.tolerance, "tolerance")
        if self.max_iterations <= 0:
            raise ValueError(
                f"max_iterations must be positive. Received {self.max_iterations}."
            )
        self._validate_market_price_bounds()

    def solve_newton(
        self,
        initial_volatility: float | None = None,
        min_volatility: float = 1e-8,
        max_volatility: float = 5.0,
    ) -> ImpliedVolatilityResult:
        """Solve implied volatility using Newton-Raphson iterations.

        Newton-Raphson is fast near the solution, but it can struggle when Vega
        is very small or the initial guess is poor.
        """
        require_positive(min_volatility, "min_volatility")
        require_positive(max_volatility, "max_volatility")
        if min_volatility >= max_volatility:
            raise ValueError("min_volatility must be less than max_volatility.")

        sigma = initial_volatility if initial_volatility is not None else self.market.volatility
        sigma = min(max(float(sigma), min_volatility), max_volatility)

        for iteration in range(1, self.max_iterations + 1):
            price = self._price_at_volatility(sigma)
            pricing_error = price - self.market_price

            if abs(pricing_error) <= self.tolerance:
                return ImpliedVolatilityResult(
                    implied_volatility=sigma,
                    iterations=iteration,
                    converged=True,
                    method="newton",
                    pricing_error=pricing_error,
                )

            vega = self._vega_at_volatility(sigma)
            if abs(vega) < 1e-12:
                break

            next_sigma = sigma - pricing_error / vega
            if not isfinite(next_sigma):
                break

            sigma = min(max(next_sigma, min_volatility), max_volatility)

        final_error = self._price_at_volatility(sigma) - self.market_price
        return ImpliedVolatilityResult(
            implied_volatility=sigma,
            iterations=self.max_iterations,
            converged=False,
            method="newton",
            pricing_error=final_error,
        )

    def solve_bisection(
        self,
        min_volatility: float = 1e-8,
        max_volatility: float = 5.0,
    ) -> ImpliedVolatilityResult:
        """Solve implied volatility using the bisection method.

        Bisection is slower than Newton-Raphson, but very robust if the target
        price is bracketed between low-vol and high-vol prices.
        """
        require_positive(min_volatility, "min_volatility")
        require_positive(max_volatility, "max_volatility")
        if min_volatility >= max_volatility:
            raise ValueError("min_volatility must be less than max_volatility.")

        low = min_volatility
        high = max_volatility
        low_error = self._price_at_volatility(low) - self.market_price
        high_error = self._price_at_volatility(high) - self.market_price

        if abs(low_error) <= self.tolerance:
            return ImpliedVolatilityResult(low, 0, True, "bisection", low_error)
        if abs(high_error) <= self.tolerance:
            return ImpliedVolatilityResult(high, 0, True, "bisection", high_error)
        if low_error * high_error > 0:
            raise ValueError(
                "Market price is not bracketed by the supplied volatility range."
            )

        mid = 0.5 * (low + high)
        mid_error = self._price_at_volatility(mid) - self.market_price

        for iteration in range(1, self.max_iterations + 1):
            mid = 0.5 * (low + high)
            mid_error = self._price_at_volatility(mid) - self.market_price

            if abs(mid_error) <= self.tolerance:
                return ImpliedVolatilityResult(
                    implied_volatility=mid,
                    iterations=iteration,
                    converged=True,
                    method="bisection",
                    pricing_error=mid_error,
                )

            if low_error * mid_error <= 0:
                high = mid
                high_error = mid_error
            else:
                low = mid
                low_error = mid_error

        return ImpliedVolatilityResult(
            implied_volatility=mid,
            iterations=self.max_iterations,
            converged=False,
            method="bisection",
            pricing_error=mid_error,
        )

    def _price_at_volatility(self, volatility: float) -> float:
        """Return the Black-Scholes price at a candidate volatility."""
        scenario_market = replace(self.market, volatility=volatility)
        return BlackScholesModel(option=self.option, market=scenario_market).price()

    def _vega_at_volatility(self, volatility: float) -> float:
        """Return Black-Scholes Vega at a candidate volatility."""
        scenario_market = replace(self.market, volatility=volatility)
        model = BlackScholesModel(option=self.option, market=scenario_market)
        return GreeksEngine(model=model).vega()

    def _validate_market_price_bounds(self) -> None:
        """Raise if the option price violates basic European no-arbitrage bounds."""
        discount_factor = BlackScholesModel(option=self.option, market=self.market).discount_factor
        discounted_strike = self.option.strike * discount_factor

        if self.option.option_type is OptionType.CALL:
            lower_bound = max(self.market.spot - discounted_strike, 0.0)
            upper_bound = self.market.spot
        elif self.option.option_type is OptionType.PUT:
            lower_bound = max(discounted_strike - self.market.spot, 0.0)
            upper_bound = discounted_strike
        else:
            raise ValueError(f"Unsupported option type: {self.option.option_type}.")

        tolerance = 1e-10
        below_lower_bound = self.market_price < lower_bound - tolerance
        above_upper_bound = self.market_price > upper_bound + tolerance
        if below_lower_bound or above_upper_bound:
            raise ValueError(
                "market_price violates no-arbitrage bounds: "
                f"price={self.market_price}, lower={lower_bound}, upper={upper_bound}."
            )


def extract_implied_volatilities(
    option_chain: pd.DataFrame,
    spot: float,
    risk_free_rate: float,
    method: str = "bisection",
    initial_volatility: float = 0.2,
) -> pd.DataFrame:
    """Extract implied volatility from an option-chain style DataFrame.

    Required columns are ``strike``, ``maturity``, ``option_type``, and
    ``market_price``. The output preserves the input rows and adds
    ``implied_volatility`` plus ``moneyness``.
    """
    require_positive(spot, "spot")
    require_non_negative(initial_volatility, "initial_volatility")

    required_columns = {"strike", "maturity", "option_type", "market_price"}
    missing = required_columns.difference(option_chain.columns)
    if missing:
        raise ValueError(f"option_chain is missing required columns: {sorted(missing)}.")

    rows: list[pd.Series] = []
    for _, row in option_chain.iterrows():
        option = EuropeanOption(
            strike=float(row["strike"]),
            maturity=float(row["maturity"]),
            option_type=OptionType(str(row["option_type"]).lower()),
        )
        market = MarketEnvironment(
            spot=spot,
            risk_free_rate=risk_free_rate,
            volatility=initial_volatility,
        )
        solver = ImpliedVolatilitySolver(
            option=option,
            market=market,
            market_price=float(row["market_price"]),
        )

        if method == "bisection":
            result = solver.solve_bisection()
        elif method == "newton":
            result = solver.solve_newton(initial_volatility=initial_volatility)
        else:
            raise ValueError("method must be either 'bisection' or 'newton'.")

        output_row = row.copy()
        output_row["implied_volatility"] = result.implied_volatility
        output_row["moneyness"] = option.strike / spot
        output_row["iv_converged"] = result.converged
        output_row["iv_iterations"] = result.iterations
        rows.append(output_row)

    return pd.DataFrame(rows).reset_index(drop=True)

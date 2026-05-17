"""Scenario and Monte Carlo stress testing for option portfolios."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt

import numpy as np
from numpy.typing import NDArray

from analytics.greeks import Greeks, GreeksEngine
from models.black_scholes import BlackScholesModel
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment
from utils.validation import require_non_negative, require_positive

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OptionPosition:
    """Portfolio position in a European option."""

    option: EuropeanOption
    quantity: float

    def __post_init__(self) -> None:
        """Validate position inputs."""
        if self.quantity == 0.0:
            raise ValueError("quantity must be non-zero.")


@dataclass(frozen=True)
class StressScenario:
    """Deterministic market shock used for scenario stress testing."""

    name: str
    spot_shock: float = 0.0
    volatility_shock: float = 0.0
    volatility_multiplier: float = 1.0
    rate_shock: float = 0.0

    def __post_init__(self) -> None:
        """Validate stress scenario inputs."""
        if not self.name:
            raise ValueError("name must be non-empty.")
        if self.spot_shock <= -1.0:
            raise ValueError("spot_shock must be greater than -100%.")
        require_non_negative(self.volatility_multiplier, "volatility_multiplier")

    def apply(self, market: MarketEnvironment) -> MarketEnvironment:
        """Apply the scenario to a base market environment."""
        stressed_spot = market.spot * (1.0 + self.spot_shock)
        stressed_volatility = market.volatility * self.volatility_multiplier + self.volatility_shock
        stressed_rate = market.risk_free_rate + self.rate_shock

        if stressed_volatility < 0.0:
            raise ValueError(
                f"scenario '{self.name}' produces negative volatility: {stressed_volatility}."
            )

        return replace(
            market,
            spot=stressed_spot,
            volatility=stressed_volatility,
            risk_free_rate=stressed_rate,
        )


@dataclass(frozen=True)
class PortfolioValuation:
    """Portfolio value and aggregate Greeks for one market environment."""

    value: float
    greeks: Greeks


@dataclass(frozen=True)
class StressResult:
    """Deterministic stress result for one scenario."""

    scenario: StressScenario
    base_value: float
    stressed_value: float
    pnl: float
    stressed_market: MarketEnvironment
    stressed_greeks: Greeks


@dataclass(frozen=True)
class MonteCarloStressResult:
    """Monte Carlo stress-test P&L distribution and tail diagnostics."""

    pnl: FloatArray
    mean_pnl: float
    std_pnl: float
    value_at_risk: float
    expected_shortfall: float
    confidence_level: float
    n_paths: int


@dataclass(frozen=True)
class StressTester:
    """Runs deterministic and simulation-based stress scenarios."""

    positions: tuple[OptionPosition, ...]
    market: MarketEnvironment

    def __post_init__(self) -> None:
        """Validate stress tester inputs."""
        if not self.positions:
            raise ValueError("positions must contain at least one option position.")

    def value_portfolio(self, market: MarketEnvironment | None = None) -> PortfolioValuation:
        """Return portfolio mark-to-model value and aggregate Greeks."""
        active_market = market or self.market
        total_value = 0.0
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0
        total_rho = 0.0

        for position in self.positions:
            model = BlackScholesModel(option=position.option, market=active_market)
            greeks = GreeksEngine(model=model).all()
            total_value += position.quantity * model.price()
            total_delta += position.quantity * greeks.delta
            total_gamma += position.quantity * greeks.gamma
            total_vega += position.quantity * greeks.vega
            total_theta += position.quantity * greeks.theta
            total_rho += position.quantity * greeks.rho

        return PortfolioValuation(
            value=float(total_value),
            greeks=Greeks(
                delta=float(total_delta),
                gamma=float(total_gamma),
                vega=float(total_vega),
                theta=float(total_theta),
                rho=float(total_rho),
            ),
        )

    def run_scenario(self, scenario: StressScenario) -> StressResult:
        """Run one deterministic market stress scenario."""
        base = self.value_portfolio()
        stressed_market = scenario.apply(self.market)
        stressed = self.value_portfolio(stressed_market)

        return StressResult(
            scenario=scenario,
            base_value=base.value,
            stressed_value=stressed.value,
            pnl=stressed.value - base.value,
            stressed_market=stressed_market,
            stressed_greeks=stressed.greeks,
        )

    def run_scenarios(self, scenarios: list[StressScenario]) -> list[StressResult]:
        """Run multiple deterministic stress scenarios."""
        if not scenarios:
            raise ValueError("scenarios must contain at least one scenario.")
        return [self.run_scenario(scenario) for scenario in scenarios]

    def monte_carlo_stress(
        self,
        n_paths: int = 10_000,
        horizon: float = 1.0 / 252.0,
        realized_volatility: float | None = None,
        volatility_multiplier: float = 1.0,
        volatility_shock: float = 0.0,
        spot_jump: float = 0.0,
        confidence_level: float = 0.95,
        seed: int | None = None,
    ) -> MonteCarloStressResult:
        """Simulate short-horizon stressed P&L distribution.

        The portfolio is revalued after a simulated spot move and volatility
        shock. Maturities are reduced by the stress horizon.
        """
        if n_paths <= 0:
            raise ValueError(f"n_paths must be positive. Received {n_paths}.")
        require_positive(horizon, "horizon")
        require_non_negative(volatility_multiplier, "volatility_multiplier")
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1.")

        base_value = self.value_portfolio().value
        simulation_volatility = (
            self.market.volatility if realized_volatility is None else realized_volatility
        )
        require_non_negative(simulation_volatility, "realized_volatility")
        stressed_volatility = self.market.volatility * volatility_multiplier + volatility_shock
        require_non_negative(stressed_volatility, "stressed_volatility")

        rng = np.random.default_rng(seed)
        shocks = rng.standard_normal(n_paths)
        log_returns = (
            (self.market.risk_free_rate - 0.5 * simulation_volatility**2) * horizon
            + simulation_volatility * sqrt(horizon) * shocks
        )
        terminal_spots = self.market.spot * (1.0 + spot_jump) * np.exp(log_returns)
        pnl = np.array(
            [
                self._value_after_horizon(
                    spot=float(spot),
                    volatility=stressed_volatility,
                    horizon=horizon,
                )
                - base_value
                for spot in terminal_spots
            ],
            dtype=np.float64,
        )

        percentile = 100.0 * (1.0 - confidence_level)
        value_at_risk = float(np.percentile(pnl, percentile))
        tail_losses = pnl[pnl <= value_at_risk]
        expected_shortfall = float(np.mean(tail_losses)) if tail_losses.size else value_at_risk

        return MonteCarloStressResult(
            pnl=pnl,
            mean_pnl=float(np.mean(pnl)),
            std_pnl=float(np.std(pnl, ddof=1)) if n_paths > 1 else 0.0,
            value_at_risk=value_at_risk,
            expected_shortfall=expected_shortfall,
            confidence_level=confidence_level,
            n_paths=n_paths,
        )

    def _value_after_horizon(self, spot: float, volatility: float, horizon: float) -> float:
        """Return portfolio value after horizon with reduced maturities."""
        value = 0.0
        market = replace(self.market, spot=spot, volatility=volatility)

        for position in self.positions:
            remaining_maturity = position.option.maturity - horizon
            if remaining_maturity <= 1e-8:
                option_value = self._payoff(position.option, spot)
            else:
                shocked_option = replace(position.option, maturity=remaining_maturity)
                option_value = BlackScholesModel(option=shocked_option, market=market).price()
            value += position.quantity * option_value

        return float(value)

    @staticmethod
    def _payoff(option: EuropeanOption, spot: float) -> float:
        """Return option terminal payoff."""
        if option.option_type is OptionType.CALL:
            return max(spot - option.strike, 0.0)
        if option.option_type is OptionType.PUT:
            return max(option.strike - spot, 0.0)
        raise ValueError(f"Unsupported option type: {option.option_type}.")


def standard_stress_scenarios() -> list[StressScenario]:
    """Return a reusable set of institutional-style stress scenarios."""
    return [
        StressScenario(name="Volatility explosion", volatility_multiplier=2.0),
        StressScenario(name="Market crash", spot_shock=-0.20, volatility_multiplier=1.75),
        StressScenario(name="Flash crash", spot_shock=-0.10, volatility_shock=0.15),
        StressScenario(name="Relief rally", spot_shock=0.12, volatility_shock=-0.05),
        StressScenario(name="Rate shock higher", rate_shock=0.01),
        StressScenario(
            name="Term-structure inversion proxy",
            volatility_shock=0.10,
            rate_shock=-0.005,
        ),
    ]

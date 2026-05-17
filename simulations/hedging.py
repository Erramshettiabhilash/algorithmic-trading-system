"""Dynamic Greeks hedging simulations and portfolio neutralization helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp

import numpy as np

from analytics.greeks import Greeks, GreeksEngine
from models.black_scholes import BlackScholesModel
from simulations.stochastic_processes import FloatArray, geometric_brownian_motion_paths, time_grid
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment
from utils.validation import require_non_negative


@dataclass(frozen=True)
class DeltaHedgePathResult:
    """Detailed accounting for one delta-hedged option path."""

    times: FloatArray
    stock_prices: FloatArray
    option_values: FloatArray
    deltas: FloatArray
    stock_positions: FloatArray
    cash_account: FloatArray
    hedge_portfolio_values: FloatArray
    transaction_costs: FloatArray
    final_pnl: float
    terminal_payoff: float


@dataclass(frozen=True)
class DeltaHedgeSummary:
    """Summary statistics across many delta-hedged paths."""

    mean_pnl: float
    std_pnl: float
    mean_absolute_error: float
    min_pnl: float
    max_pnl: float
    n_paths: int


@dataclass(frozen=True)
class HedgePortfolioLeg:
    """Option position and Greeks used for portfolio neutralization."""

    quantity: float
    greeks: Greeks


@dataclass(frozen=True)
class HedgingSimulator:
    """Simulates discrete Delta hedging for a European option position.

    Args:
        option: Option contract being hedged.
        market: Initial market environment and hedge-model volatility.
        option_quantity: Option position. Positive is long, negative is short.
        transaction_cost_rate: Proportional cost paid on stock notional traded.
    """

    option: EuropeanOption
    market: MarketEnvironment
    option_quantity: float = -1.0
    transaction_cost_rate: float = 0.0

    def __post_init__(self) -> None:
        """Validate hedge simulator inputs."""
        require_non_negative(self.transaction_cost_rate, "transaction_cost_rate")
        if self.option_quantity == 0.0:
            raise ValueError("option_quantity must be non-zero.")

    def simulate_price_paths(
        self,
        n_paths: int,
        steps: int,
        seed: int | None = None,
        realized_volatility: float | None = None,
        drift: float | None = None,
    ) -> FloatArray:
        """Simulate stock paths for hedging experiments."""
        volatility = self.market.volatility if realized_volatility is None else realized_volatility
        path_drift = self.market.risk_free_rate if drift is None else drift
        return geometric_brownian_motion_paths(
            spot=self.market.spot,
            drift=path_drift,
            volatility=volatility,
            maturity=self.option.maturity,
            steps=steps,
            n_paths=n_paths,
            seed=seed,
        )

    def simulate_delta_hedge_path(self, stock_prices: FloatArray) -> DeltaHedgePathResult:
        """Simulate discrete Delta hedging along one supplied stock path."""
        if stock_prices.ndim != 1:
            raise ValueError("stock_prices must be a one-dimensional path.")
        if stock_prices.size < 2:
            raise ValueError("stock_prices must contain at least two prices.")
        if np.any(stock_prices <= 0.0):
            raise ValueError("stock_prices must be strictly positive.")

        steps = stock_prices.size - 1
        times = time_grid(maturity=self.option.maturity, steps=steps)
        dt = self.option.maturity / steps

        option_values = np.zeros(stock_prices.size, dtype=np.float64)
        deltas = np.zeros(stock_prices.size, dtype=np.float64)
        stock_positions = np.zeros(stock_prices.size, dtype=np.float64)
        cash_account = np.zeros(stock_prices.size, dtype=np.float64)
        hedge_values = np.zeros(stock_prices.size, dtype=np.float64)
        transaction_costs = np.zeros(stock_prices.size, dtype=np.float64)

        initial_model = self._model_at(stock=float(stock_prices[0]), maturity=self.option.maturity)
        option_values[0] = initial_model.price()
        deltas[0] = GreeksEngine(model=initial_model).delta()
        stock_positions[0] = -self.option_quantity * deltas[0]
        initial_trade_cost = self._transaction_cost(stock_positions[0], float(stock_prices[0]))
        transaction_costs[0] = initial_trade_cost
        cash_account[0] = (
            -self.option_quantity * option_values[0]
            - stock_positions[0] * stock_prices[0]
            - initial_trade_cost
        )
        hedge_values[0] = cash_account[0] + stock_positions[0] * stock_prices[0]

        for step in range(1, steps):
            cash_account[step] = cash_account[step - 1] * exp(self.market.risk_free_rate * dt)
            remaining_maturity = self.option.maturity - times[step]
            model = self._model_at(stock=float(stock_prices[step]), maturity=remaining_maturity)
            option_values[step] = model.price()
            deltas[step] = GreeksEngine(model=model).delta()

            target_stock_position = -self.option_quantity * deltas[step]
            trade_size = target_stock_position - stock_positions[step - 1]
            trade_cost = self._transaction_cost(trade_size, float(stock_prices[step]))

            transaction_costs[step] = trade_cost
            cash_account[step] -= trade_size * stock_prices[step] + trade_cost
            stock_positions[step] = target_stock_position
            hedge_values[step] = cash_account[step] + stock_positions[step] * stock_prices[step]

        cash_account[-1] = cash_account[-2] * exp(self.market.risk_free_rate * dt)
        stock_positions[-1] = stock_positions[-2]
        terminal_payoff = self._payoff(float(stock_prices[-1]))
        option_values[-1] = terminal_payoff
        deltas[-1] = 0.0
        hedge_values[-1] = cash_account[-1] + stock_positions[-1] * stock_prices[-1]
        final_pnl = hedge_values[-1] + self.option_quantity * terminal_payoff

        return DeltaHedgePathResult(
            times=times,
            stock_prices=stock_prices,
            option_values=option_values,
            deltas=deltas,
            stock_positions=stock_positions,
            cash_account=cash_account,
            hedge_portfolio_values=hedge_values,
            transaction_costs=transaction_costs,
            final_pnl=float(final_pnl),
            terminal_payoff=float(terminal_payoff),
        )

    def simulate_delta_hedging(
        self,
        n_paths: int,
        steps: int,
        seed: int | None = None,
        realized_volatility: float | None = None,
        drift: float | None = None,
    ) -> tuple[list[DeltaHedgePathResult], DeltaHedgeSummary]:
        """Simulate Delta hedging across many stock paths."""
        if n_paths <= 0:
            raise ValueError(f"n_paths must be positive. Received {n_paths}.")
        if steps <= 0:
            raise ValueError(f"steps must be positive. Received {steps}.")

        paths = self.simulate_price_paths(
            n_paths=n_paths,
            steps=steps,
            seed=seed,
            realized_volatility=realized_volatility,
            drift=drift,
        )
        results = [self.simulate_delta_hedge_path(path) for path in paths]
        pnl = np.array([result.final_pnl for result in results], dtype=np.float64)
        summary = DeltaHedgeSummary(
            mean_pnl=float(np.mean(pnl)),
            std_pnl=float(np.std(pnl, ddof=1)) if n_paths > 1 else 0.0,
            mean_absolute_error=float(np.mean(np.abs(pnl))),
            min_pnl=float(np.min(pnl)),
            max_pnl=float(np.max(pnl)),
            n_paths=n_paths,
        )
        return results, summary

    def _model_at(self, stock: float, maturity: float) -> BlackScholesModel:
        """Build a Black-Scholes model at a path node."""
        node_option = replace(self.option, maturity=maturity)
        node_market = replace(self.market, spot=stock)
        return BlackScholesModel(option=node_option, market=node_market)

    def _payoff(self, stock: float) -> float:
        """Return terminal option payoff."""
        if self.option.option_type is OptionType.CALL:
            return max(stock - self.option.strike, 0.0)
        if self.option.option_type is OptionType.PUT:
            return max(self.option.strike - stock, 0.0)
        raise ValueError(f"Unsupported option type: {self.option.option_type}.")

    def _transaction_cost(self, trade_size: float, stock: float) -> float:
        """Return proportional stock transaction cost for one hedge trade."""
        return abs(trade_size) * stock * self.transaction_cost_rate


def delta_neutral_stock_position(option_position: float, option_delta: float) -> float:
    """Return stock quantity required to make an option position Delta-neutral."""
    return -option_position * option_delta


def gamma_vega_neutral_option_positions(
    target: HedgePortfolioLeg,
    gamma_hedge: Greeks,
    vega_hedge: Greeks,
) -> tuple[float, float]:
    """Return two option quantities that neutralize target Gamma and Vega.

    The returned quantities correspond to ``gamma_hedge`` and ``vega_hedge``.
    """
    exposure = np.array(
        [
            -target.quantity * target.greeks.gamma,
            -target.quantity * target.greeks.vega,
        ],
        dtype=np.float64,
    )
    hedge_matrix = np.array(
        [
            [gamma_hedge.gamma, vega_hedge.gamma],
            [gamma_hedge.vega, vega_hedge.vega],
        ],
        dtype=np.float64,
    )

    try:
        quantities = np.linalg.solve(hedge_matrix, exposure)
    except np.linalg.LinAlgError as exc:
        raise ValueError("hedge instruments cannot span Gamma and Vega exposures.") from exc

    return float(quantities[0]), float(quantities[1])

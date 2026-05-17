"""Run an end-to-end options analytics demo and save example outputs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.implied_volatility import extract_implied_volatilities
from analytics.market_making import InventoryState, MarketMakingEngine
from analytics.reporting import option_chain_analytics
from analytics.stress_testing import OptionPosition, StressTester, standard_stress_scenarios
from analytics.volatility_surface import VolatilitySurfaceAnalyzer
from models.black_scholes import BlackScholesModel
from models.sabr import calibrate_sabr_smile
from simulations.hedging import HedgingSimulator
from simulations.monte_carlo import MonteCarloEngine
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment
from visualization.charts import (
    plot_greeks_vs_spot,
    plot_hedging_pnl,
    plot_option_payoff,
    plot_option_price_heatmap,
    plot_payoff_distribution,
    plot_sabr_fit,
    plot_simulated_paths,
    plot_stress_pnl_distribution,
    plot_stress_scenario_pnl,
    plot_volatility_smile,
    plot_volatility_surface,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "examples"


def main() -> None:
    """Generate a compact, reproducible analytics showcase."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    market = MarketEnvironment(spot=100.0, risk_free_rate=0.05, volatility=0.20)
    call = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
    put = EuropeanOption(strike=95.0, maturity=1.0, option_type=OptionType.PUT)
    spots = np.linspace(60.0, 140.0, 81)
    volatilities = np.linspace(0.10, 0.50, 41)

    option_chain = [
        EuropeanOption(strike=strike, maturity=1.0, option_type=OptionType.CALL)
        for strike in (90.0, 100.0, 110.0)
    ]
    analytics_table = option_chain_analytics(option_chain, market)
    analytics_table.to_csv(OUTPUT_DIR / "option_chain_analytics.csv", index=False)

    call_model = BlackScholesModel(option=call, market=market)
    mc_engine = MonteCarloEngine(option=call, market=market, n_paths=20_000, steps=252, seed=7)
    mc_result = mc_engine.price()
    paths = mc_engine.simulate_paths()
    payoffs = mc_engine.payoff(paths[:, -1])

    surface_data = pd.read_csv(PROJECT_ROOT / "data" / "sample_vol_surface.csv")
    surface = VolatilitySurfaceAnalyzer(data=surface_data, spot=market.spot)
    one_year_smile = surface.smile(maturity=1.0)
    surface_grid = surface.surface_grid()

    sabr_result = calibrate_sabr_smile(
        forward=market.spot,
        strikes=one_year_smile["strike"].to_numpy(dtype=np.float64),
        market_volatilities=one_year_smile["implied_volatility"].to_numpy(dtype=np.float64),
        maturity=1.0,
        beta=0.5,
        initial_guess=(2.0, -0.2, 0.5),
    )
    sabr_vols = sabr_result.model.smile(
        forward=market.spot,
        strikes=one_year_smile["strike"].to_numpy(dtype=np.float64),
        maturity=1.0,
    )

    market_prices = []
    for _, row in one_year_smile.iterrows():
        option = EuropeanOption(
            strike=float(row["strike"]),
            maturity=float(row["maturity"]),
            option_type=OptionType(str(row["option_type"]).lower()),
        )
        row_market = MarketEnvironment(
            spot=market.spot,
            risk_free_rate=market.risk_free_rate,
            volatility=float(row["implied_volatility"]),
        )
        market_prices.append(BlackScholesModel(option=option, market=row_market).price())

    iv_input = one_year_smile[["strike", "maturity", "option_type"]].copy()
    iv_input["market_price"] = market_prices
    extracted_iv = extract_implied_volatilities(
        option_chain=iv_input,
        spot=market.spot,
        risk_free_rate=market.risk_free_rate,
    )
    extracted_iv.to_csv(OUTPUT_DIR / "extracted_implied_volatility.csv", index=False)

    hedge_simulator = HedgingSimulator(option=call, market=market, option_quantity=-1.0)
    hedge_path = hedge_simulator.simulate_price_paths(n_paths=1, steps=52, seed=3)[0]
    hedge_result = hedge_simulator.simulate_delta_hedge_path(hedge_path)

    stress_tester = StressTester(
        positions=(
            OptionPosition(option=call, quantity=1.0),
            OptionPosition(option=put, quantity=-1.0),
        ),
        market=market,
    )
    stress_results = stress_tester.run_scenarios(standard_stress_scenarios())
    stress_table = pd.DataFrame(
        [
            {
                "scenario": result.scenario.name,
                "spot": result.stressed_market.spot,
                "volatility": result.stressed_market.volatility,
                "risk_free_rate": result.stressed_market.risk_free_rate,
                "pnl": result.pnl,
                "delta": result.stressed_greeks.delta,
                "gamma": result.stressed_greeks.gamma,
                "vega": result.stressed_greeks.vega,
            }
            for result in stress_results
        ]
    )
    stress_table.to_csv(OUTPUT_DIR / "stress_scenarios.csv", index=False)
    mc_stress = stress_tester.monte_carlo_stress(
        n_paths=5_000,
        horizon=5.0 / 252.0,
        realized_volatility=0.40,
        volatility_multiplier=1.5,
        seed=11,
    )

    market_maker = MarketMakingEngine(option=call, market=market)
    quote = market_maker.quote(InventoryState(option_position=15.0, cash=0.0))
    quote_payload = asdict(quote)
    (OUTPUT_DIR / "market_making_quote.json").write_text(
        json.dumps(quote_payload, indent=2),
        encoding="utf-8",
    )

    plot_option_payoff(
        terminal_prices=spots,
        option=call,
        premium=call_model.price(),
        output_path=str(OUTPUT_DIR / "payoff_diagram.png"),
    )
    plot_greeks_vs_spot(
        spots=spots,
        option=call,
        market=market,
        output_path=str(OUTPUT_DIR / "greeks_vs_spot.png"),
    )
    plot_option_price_heatmap(
        spots=spots,
        volatilities=volatilities,
        option=call,
        market=market,
        output_path=str(OUTPUT_DIR / "price_heatmap.png"),
    )
    plot_simulated_paths(
        paths=paths[:100],
        maturity=call.maturity,
        output_path=str(OUTPUT_DIR / "monte_carlo_paths.png"),
    )
    plot_payoff_distribution(
        payoffs=payoffs,
        output_path=str(OUTPUT_DIR / "payoff_distribution.png"),
    )
    plot_volatility_smile(
        strikes=one_year_smile["strike"].to_numpy(dtype=np.float64),
        implied_volatilities=one_year_smile["implied_volatility"].to_numpy(dtype=np.float64),
        output_path=str(OUTPUT_DIR / "volatility_smile.png"),
    )
    plot_volatility_surface(surface_grid, output_path=str(OUTPUT_DIR / "volatility_surface.html"))
    plot_sabr_fit(
        strikes=one_year_smile["strike"].to_numpy(dtype=np.float64),
        market_volatilities=one_year_smile["implied_volatility"].to_numpy(dtype=np.float64),
        sabr_volatilities=sabr_vols,
        output_path=str(OUTPUT_DIR / "sabr_fit.png"),
    )
    plot_hedging_pnl(hedge_result, output_path=str(OUTPUT_DIR / "hedging_pnl.png"))
    plot_stress_scenario_pnl(stress_results, output_path=str(OUTPUT_DIR / "stress_scenarios.png"))
    plot_stress_pnl_distribution(
        mc_stress.pnl,
        value_at_risk=mc_stress.value_at_risk,
        expected_shortfall=mc_stress.expected_shortfall,
        output_path=str(OUTPUT_DIR / "stress_pnl_distribution.png"),
    )

    summary = {
        "black_scholes_call": call_model.price(),
        "monte_carlo_call": mc_result.price,
        "monte_carlo_standard_error": mc_result.standard_error,
        "sabr_rmse": sabr_result.rmse,
        "hedging_final_pnl": hedge_result.final_pnl,
        "stress_var_95": mc_stress.value_at_risk,
        "stress_expected_shortfall_95": mc_stress.expected_shortfall,
        "market_making_bid": quote.bid,
        "market_making_ask": quote.ask,
    }
    (OUTPUT_DIR / "demo_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote demo outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

# Step 11: Stress Testing

Stress testing asks:

```text
What happens to the portfolio when normal market assumptions fail?
```

Pricing and Greeks describe current risk. Stress testing describes damage under
extreme but plausible market moves.

## 1. Why Stress Testing Matters

Options are nonlinear. A small spot move can be manageable, while a large move
can produce losses that Greeks estimated at the starting point did not fully
capture.

Stress tests are used for:

- Crash risk
- Volatility shocks
- Liquidity planning
- Hedging breakdown analysis
- Tail-loss reporting
- Risk-limit governance

## 2. Deterministic Scenarios

A deterministic scenario applies explicit shocks:

```text
spot shock
volatility shock
volatility multiplier
rate shock
```

Example:

```text
Market crash:
spot -20%
volatility x 1.75
```

The portfolio is revalued under the stressed market.

```text
P&L = stressed value - base value
```

## 3. Volatility Explosion

A volatility explosion scenario increases implied volatility sharply.

Long vanilla options usually benefit from this because Vega is positive.
Short-option portfolios usually lose.

## 4. Market Crash

Equity crashes often combine:

```text
spot down
volatility up
```

This is dangerous for short puts and short-Gamma books.

## 5. Flash Crash

A flash crash is a fast spot gap, often with implied volatility repricing.

Delta hedging can break down because the hedge cannot be rebalanced
continuously through a gap.

## 6. Term-Structure Inversion Proxy

A real volatility term-structure inversion means short-dated implied volatility
rises above long-dated implied volatility.

Our current single-option market environment does not yet hold a full curve, so
Step 11 includes a proxy scenario:

```text
volatility up
rates down
```

Later, this can be extended to shock each maturity bucket in the volatility
surface separately.

## 7. Monte Carlo Stress

Monte Carlo stress simulates many short-horizon stressed spot moves, then
revalues the portfolio after the horizon.

The output includes:

- Mean P&L
- P&L standard deviation
- Value at Risk
- Expected shortfall
- Full P&L distribution

Value at Risk:

```text
5th percentile P&L for 95% confidence
```

Expected shortfall:

```text
average P&L conditional on being worse than VaR
```

Expected shortfall is often more informative because it looks inside the tail,
not just at one percentile.

## 8. Code Entry Point

The implementation lives in:

```text
analytics/stress_testing.py
```

Main objects:

```text
OptionPosition
StressScenario
StressTester
StressResult
MonteCarloStressResult
standard_stress_scenarios
```

Basic usage:

```python
from analytics.stress_testing import OptionPosition, StressTester, standard_stress_scenarios
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment

option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.PUT)
market = MarketEnvironment(spot=100.0, risk_free_rate=0.03, volatility=0.2)

tester = StressTester(
    positions=(OptionPosition(option=option, quantity=-1.0),),
    market=market,
)

scenario_results = tester.run_scenarios(standard_stress_scenarios())
mc_result = tester.monte_carlo_stress(
    n_paths=10_000,
    horizon=5.0 / 252.0,
    realized_volatility=0.45,
    volatility_multiplier=1.5,
    spot_jump=-0.03,
    seed=7,
)
```

## 9. Professional Interpretation

Stress testing is not about predicting the exact crisis. It is about discovering
portfolio fragility before the market does.

Useful questions:

- Which scenario creates the largest loss?
- Do Greeks increase or flip under stress?
- Does a short-vol book survive a volatility shock?
- How large is expected shortfall?
- Does hedging still work after a gap?


# Step 12: Visualization & Analytics

Step 12 turns the pricing engines into research outputs.

Professional quant work is not finished when a function returns a number. The
result needs to be inspected, explained, charted, and compared across scenarios.

## 1. Why Visualization Matters

Good charts help answer:

- Is the payoff shape correct?
- Do Greeks behave sensibly across spot?
- Does Monte Carlo converge toward theory?
- Is the implied-volatility smile smooth?
- Where does the stress loss come from?
- Is the hedging error centered or biased?

Visualization is a model validation tool, not decoration.

## 2. Analytics Tables

The module:

```text
analytics/reporting.py
```

adds:

```text
option_analytics_snapshot
option_chain_analytics
```

These return clean price and Greeks summaries before plotting.

Example:

```python
from analytics.reporting import option_chain_analytics
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment

market = MarketEnvironment(spot=100.0, risk_free_rate=0.05, volatility=0.2)
options = [
    EuropeanOption(strike=90.0, maturity=1.0, option_type=OptionType.CALL),
    EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL),
    EuropeanOption(strike=110.0, maturity=1.0, option_type=OptionType.CALL),
]

table = option_chain_analytics(options=options, market=market)
print(table)
```

## 3. Payoff Diagrams

Payoff diagrams show terminal option economics.

```python
import numpy as np

from visualization.charts import plot_option_payoff

terminal_prices = np.linspace(50.0, 150.0, 101)

plot_option_payoff(
    terminal_prices=terminal_prices,
    option=options[1],
    premium=10.0,
)
```

For interviews, payoff diagrams are useful because they connect contract
definition to risk intuition.

## 4. Greeks Curves

Greeks are not constants. They change with spot, maturity, and volatility.

```python
from visualization.charts import plot_greeks_vs_spot

spots = np.linspace(50.0, 150.0, 101)
plot_greeks_vs_spot(spots=spots, option=options[1], market=market)
```

This helps explain why hedges must be rebalanced.

## 5. Price Sensitivity Heatmaps

The heatmap answers:

```text
How does option price change when spot and volatility move together?
```

```python
from visualization.charts import plot_option_price_heatmap

plot_option_price_heatmap(
    spots=np.linspace(70.0, 130.0, 31),
    volatilities=np.linspace(0.10, 0.50, 21),
    option=options[1],
    market=market,
)
```

This is a compact sensitivity analysis view.

## 6. Volatility Visuals

Already available:

```text
plot_volatility_smile
plot_term_structure
plot_volatility_surface
plot_sabr_fit
```

These are used to inspect implied-volatility structure and SABR calibration.

## 7. Simulation Visuals

Already available:

```text
plot_simulated_paths
plot_payoff_distribution
```

These help diagnose Monte Carlo behavior before trusting the estimated price.

## 8. Hedging And Stress Visuals

Already available:

```text
plot_hedging_pnl
plot_hedging_error_distribution
plot_stress_scenario_pnl
plot_stress_pnl_distribution
```

These transform dynamic hedging and stress testing into risk reports.

## 9. Publication-Quality Defaults

The chart module applies:

- Consistent figure sizes
- Clean grid style
- Clear axis labels
- Saved-file support
- Non-interactive Matplotlib backend for reproducible automation

This matters because the final project should be GitHub-ready and runnable in
headless environments such as CI or remote research machines.


# Step 9: Dynamic Greeks Hedging

Dynamic hedging means repeatedly adjusting a hedge as the underlying moves and
the option Greeks change.

The most common first hedge is Delta hedging.

## 1. Delta Hedging

If an option has Delta:

```text
Delta = dV / dS
```

then an option position has directional exposure:

```text
option quantity * Delta
```

The stock hedge needed to neutralize Delta is:

```text
stock hedge = -option quantity * Delta
```

Example:

```text
short 1 call
call Delta = 0.60
stock hedge = +0.60 shares
```

## 2. Why Rebalancing Is Needed

Delta changes when spot changes. That change is Gamma.

```text
Gamma = d Delta / dS
```

High Gamma means the hedge becomes stale quickly. Market makers rebalance
because the option is not a fixed share equivalent.

## 3. Hedge P&L

The simulation tracks:

- Option premium
- Stock hedge
- Cash account
- Interest accrual
- Rebalancing trades
- Transaction costs
- Final option payoff
- Final hedge error

If the model assumptions are correct and rebalancing is frequent, average hedge
error should shrink. With discrete rebalancing, transaction costs, jumps, or
wrong volatility, hedge errors remain.

## 4. Gamma And Vega Neutrality

Delta hedging uses stock. Stock has:

```text
Delta = 1
Gamma = 0
Vega = 0
```

So stock can neutralize Delta, but not Gamma or Vega.

To neutralize Gamma and Vega, we need other options:

```text
target gamma + q1 hedge1 gamma + q2 hedge2 gamma = 0
target vega  + q1 hedge1 vega  + q2 hedge2 vega  = 0
```

The project solves this two-instrument hedge as a linear system.

## 5. Code Entry Point

The implementation lives in:

```text
simulations/hedging.py
```

Main objects:

```text
HedgingSimulator
DeltaHedgePathResult
DeltaHedgeSummary
HedgePortfolioLeg
```

Basic usage:

```python
from simulations.hedging import HedgingSimulator
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment

option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
market = MarketEnvironment(spot=100.0, risk_free_rate=0.02, volatility=0.2)

simulator = HedgingSimulator(option=option, market=market, option_quantity=-1.0)
results, summary = simulator.simulate_delta_hedging(
    n_paths=100,
    steps=52,
    seed=7,
)

print(summary.mean_pnl)
print(summary.mean_absolute_error)
```

## 6. Practical Limitations

Real hedging is messy:

- Rebalancing is discrete, not continuous.
- Trading costs matter.
- Volatility is not constant.
- Markets gap.
- Liquidity can disappear.
- Model Greeks can be wrong.

This is why hedging is risk management, not risk elimination.


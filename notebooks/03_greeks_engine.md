# Step 4: Greeks Engine

Option prices are useful, but desks manage risk through sensitivities. Greeks
answer the practical question:

```text
What happens to my option value if a market input moves?
```

The implementation lives in:

```text
analytics/greeks.py
```

## 1. Delta

Delta measures sensitivity to the underlying price:

```text
Delta = dV / dS
```

Black-Scholes formulas:

```text
Call Delta = N(d1)
Put Delta  = N(d1) - 1
```

Trading intuition:

- A call delta of 0.64 means the call behaves roughly like 0.64 shares.
- A put delta of -0.36 means the put gains when the underlying falls.
- Delta is the first hedge a market maker usually manages.

## 2. Gamma

Gamma measures how fast Delta changes:

```text
Gamma = d2V / dS2
```

Formula:

```text
Gamma = phi(d1) / (S sigma sqrt(T))
```

Where ``phi`` is the standard normal density.

Trading intuition:

- High Gamma means Delta changes quickly.
- At-the-money short-dated options often have high Gamma.
- Market makers short Gamma must rebalance frequently when spot moves.

## 3. Vega

Vega measures sensitivity to volatility:

```text
Vega = dV / d sigma
```

Formula:

```text
Vega = S phi(d1) sqrt(T)
```

Trading intuition:

- Long options usually have positive Vega.
- If implied volatility rises, long vanilla calls and puts usually gain value.
- Vega is central to volatility trading because traders often care more about
  implied volatility than directional stock forecasts.

In code, Vega is returned per 1.00 volatility move. Divide by 100 for a 1
volatility-point move. For example, 20% to 21% is a 0.01 move.

## 4. Theta

Theta measures sensitivity to time passing:

```text
Theta = dV / dt
```

In this project we report annual Black-Scholes Theta:

```text
Call Theta =
    -S phi(d1) sigma / (2 sqrt(T))
    - r K exp(-rT) N(d2)

Put Theta =
    -S phi(d1) sigma / (2 sqrt(T))
    + r K exp(-rT) N(-d2)
```

Trading intuition:

- Long options often lose value as time passes.
- Short option positions collect Theta but take Gamma and Vega risk.
- Theta is not free money; it is compensation for carrying convexity risk.

## 5. Rho

Rho measures sensitivity to the risk-free rate:

```text
Rho = dV / dr
```

Formulas:

```text
Call Rho = K T exp(-rT) N(d2)
Put Rho  = -K T exp(-rT) N(-d2)
```

Trading intuition:

- Calls generally gain when rates rise.
- Puts generally lose when rates rise.
- Rho matters more for long-dated options.

In code, Rho is returned per 1.00 rate move. Divide by 100 for a 1 percentage
point rate move.

## 6. Benchmark Example

Inputs:

```text
S = 100
K = 100
T = 1
r = 0.05
sigma = 0.20
```

Call Greeks:

```text
Delta = 0.6368
Gamma = 0.0188
Vega  = 37.5240
Theta = -6.4140
Rho   = 53.2325
```

Put Greeks:

```text
Delta = -0.3632
Gamma = 0.0188
Vega  = 37.5240
Theta = -1.6579
Rho   = -41.8905
```

## 7. Hedging Interpretation

A market maker who sells one call with Delta 0.64 is short 0.64 deltas. A simple
Delta hedge would buy 0.64 shares. If spot moves, Gamma changes Delta, so the
hedge must be updated.

This is why Greeks are dynamic risk measures, not static labels.

## 8. Visualization

The helper below plots Greeks across spot prices:

```python
import numpy as np

from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment
from visualization.charts import plot_greeks_vs_spot

option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
market = MarketEnvironment(spot=100.0, risk_free_rate=0.05, volatility=0.2)
spots = np.linspace(50.0, 150.0, 101)

plot_greeks_vs_spot(spots=spots, option=option, market=market)
```

This is the first step toward professional risk dashboards.


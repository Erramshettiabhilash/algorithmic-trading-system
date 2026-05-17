# Step 7: Volatility Surface Module

In Step 6, we inverted one option price into one implied volatility. In Step 7,
we organize many implied volatilities across strikes and maturities.

That object is the volatility surface.

```text
implied volatility = f(strike, maturity)
```

## 1. Smile

A volatility smile is one maturity slice:

```text
strike -> implied volatility
```

If Black-Scholes were perfectly true, every strike at the same maturity would
have the same implied volatility. Real markets do not behave that way.

Common reasons:

- Crash risk
- Demand for downside protection
- Leverage effects
- Jumps and fat tails
- Supply and demand across strikes

## 2. Skew

Skew describes the slope of implied volatility across strike or moneyness.

Moneyness:

```text
moneyness = strike / spot
```

Equity index options often show negative skew:

```text
lower strikes -> higher implied volatility
higher strikes -> lower implied volatility
```

Why OTM puts often have higher IV:

- Investors buy crash protection.
- Dealers who sell downside puts need compensation for gap risk.
- Equity markets tend to fall faster than they rise.
- Volatility often increases during selloffs.

## 3. Term Structure

The volatility term structure is one strike or moneyness slice across maturity:

```text
maturity -> implied volatility
```

It answers:

- Is near-term event risk elevated?
- Is long-dated uncertainty high?
- Is the market pricing a short-term volatility shock?

## 4. Surface

The full surface combines smiles and term structures:

```text
rows: maturities
columns: strikes
values: implied volatilities
```

A complete rectangular grid is useful for:

- 3D visualization
- Interpolation
- Risk reports
- Model calibration
- SABR fitting in Step 8

## 5. Code Entry Point

The implementation lives in:

```text
analytics/volatility_surface.py
```

Main class:

```python
VolatilitySurfaceAnalyzer
```

It provides:

```text
available_maturities()
available_strikes()
smile(maturity)
term_structure_by_strike(strike)
atm_term_structure()
skew(maturity)
surface_grid()
```

## 6. Basic Usage

```python
import pandas as pd

from analytics.volatility_surface import VolatilitySurfaceAnalyzer
from visualization.charts import (
    plot_term_structure,
    plot_volatility_smile,
    plot_volatility_surface,
)

surface_data = pd.read_csv("data/sample_vol_surface.csv")

analyzer = VolatilitySurfaceAnalyzer(data=surface_data, spot=100.0)

smile = analyzer.smile(maturity=1.0)
skew = analyzer.skew(maturity=1.0)
atm_term = analyzer.atm_term_structure()
grid = analyzer.surface_grid()

plot_volatility_smile(
    strikes=smile["strike"].to_numpy(),
    implied_volatilities=smile["implied_volatility"].to_numpy(),
)

plot_term_structure(
    maturities=atm_term["maturity"].to_numpy(),
    implied_volatilities=atm_term["implied_volatility"].to_numpy(),
)

fig = plot_volatility_surface(grid)
```

## 7. Institutional Interpretation

A volatility surface is a map of market fear, demand, and convexity pricing.

Traders use it to answer questions such as:

- Are downside puts expensive or cheap?
- Is event volatility concentrated in short maturities?
- Is the surface smooth enough to quote from?
- Which strikes carry the most skew risk?
- Does a model like SABR fit the observed smile?

## 8. Why This Matters Before SABR

SABR calibration needs market smile data. Before fitting a stochastic-volatility
model, we need clean volatility slices:

```text
strike, maturity, implied volatility
```

That is exactly what this module prepares.


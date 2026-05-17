# Step 8: SABR Stochastic Volatility Model

SABR is a stochastic-volatility model used to fit and interpolate implied
volatility smiles.

Black-Scholes assumes one constant volatility. Market smiles show that this is
not enough. SABR gives us a compact parametric way to describe the smile.

## 1. SABR Dynamics

The SABR model is usually written for a forward price:

```text
dF_t = alpha_t F_t^beta dW_t
d alpha_t = nu alpha_t dZ_t
corr(dW_t, dZ_t) = rho
```

Where:

- ``F_t`` is the forward price.
- ``alpha_t`` is the stochastic volatility level.
- ``beta`` controls the backbone shape.
- ``rho`` controls correlation between price and volatility shocks.
- ``nu`` is volatility of volatility.

## 2. Parameter Intuition

``alpha``

The starting volatility level. When ``beta < 1``, alpha is not exactly the ATM
Black-Scholes volatility; it is scaled by the forward level.

``beta``

The backbone parameter.

```text
beta = 0   normal-like dynamics
beta = 1   lognormal-like dynamics
```

In practice, beta is often fixed during calibration to reduce instability.

``rho``

Correlation between forward moves and volatility moves.

Negative ``rho`` creates equity-style downside skew:

```text
market falls -> volatility rises
```

``nu``

Volatility of volatility. Higher ``nu`` creates more smile curvature.

## 3. Hagan Implied Volatility Approximation

Instead of simulating the SABR stochastic differential equations every time, we
use the Hagan lognormal implied-volatility approximation.

For ``F != K``:

```text
sigma_SABR(F, K) =
    [alpha / D(F, K)] * [z / x(z)] * time_adjustment
```

The exact implementation is in:

```text
models/sabr.py
```

There is a separate ATM formula because the general expression has a removable
singularity when ``F = K``.

## 4. Calibration

Given market smile data:

```text
strike_i, market_iv_i
```

We choose SABR parameters to minimize:

```text
model_iv_i - market_iv_i
```

In this project, calibration fits:

```text
alpha, rho, nu
```

while ``beta`` is fixed.

Why fix beta?

All four SABR parameters can be hard to identify from one smile. Fixing beta is
a common professional choice because it stabilizes calibration.

## 5. Code Entry Point

```python
import numpy as np

from models.sabr import SABRModel, calibrate_sabr_smile

strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
market_vols = np.array([0.2374, 0.2286, 0.2200, 0.2126, 0.2064])

result = calibrate_sabr_smile(
    forward=100.0,
    strikes=strikes,
    market_volatilities=market_vols,
    maturity=1.0,
    beta=0.5,
    initial_guess=(2.0, -0.2, 0.5),
)

fitted_model = result.model
sabr_vols = fitted_model.smile(forward=100.0, strikes=strikes, maturity=1.0)

print(result.rmse)
print(fitted_model)
```

## 6. Visual Diagnostic

```python
from visualization.charts import plot_sabr_fit

plot_sabr_fit(
    strikes=strikes,
    market_volatilities=market_vols,
    sabr_volatilities=sabr_vols,
)
```

A good first diagnostic is simple:

```text
Do the SABR fitted vols pass through the market smile points?
```

Then check:

- Calibration RMSE
- Parameter reasonableness
- Smoothness between strikes
- Stability across maturities

## 7. SABR Vs Black-Scholes

Black-Scholes:

```text
one volatility for all strikes
```

SABR:

```text
volatility changes by strike according to alpha, beta, rho, nu
```

This makes SABR more useful for smile-aware pricing, interpolation, and risk.

## 8. Professional Caveats

SABR is useful, but not magic.

- Calibration can be unstable with sparse or noisy market data.
- Bad initial guesses can lead to bad local fits.
- Extrapolation outside quoted strikes can be dangerous.
- Arbitrage-free surface construction needs additional care.

In a real desk toolkit, SABR is one layer inside a broader volatility-surface
workflow, not the whole system.


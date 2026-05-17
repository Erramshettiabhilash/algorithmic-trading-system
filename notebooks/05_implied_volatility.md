# Step 6: Implied Volatility Engine

Implied volatility is the volatility input that makes Black-Scholes match an
observed market option price.

In pricing mode, we ask:

```text
Given sigma, what is the option price?
```

In implied-volatility mode, we ask:

```text
Given the market price, what sigma explains it?
```

## 1. Why Implied Volatility Matters

Options traders usually compare options by implied volatility, not raw price.

Raw option prices are hard to compare because they depend on:

- Spot
- Strike
- Maturity
- Rates
- Call or put type

Implied volatility converts the option price into a common volatility language.

## 2. The Inversion Problem

Black-Scholes price:

```text
V = BS(S, K, T, r, sigma)
```

Implied volatility solves:

```text
BS(S, K, T, r, sigma_implied) - MarketPrice = 0
```

This is a root-finding problem.

## 3. Newton-Raphson Method

Newton-Raphson updates volatility using the pricing error and Vega:

```text
sigma_next = sigma - [BS(sigma) - MarketPrice] / Vega
```

Why Vega appears:

Vega is the derivative of option price with respect to volatility. It tells the
solver how much the price should move when volatility changes.

Strength:

- Fast when the initial guess is reasonable.

Weakness:

- Can struggle when Vega is tiny.
- Can jump outside sensible volatility ranges.
- Less reliable for deep ITM/OTM or very short-dated options.

## 4. Bisection Method

Bisection brackets the solution between low and high volatilities:

```text
low_vol <= sigma_implied <= high_vol
```

Then it repeatedly cuts the interval in half.

Strength:

- Robust.
- Easy to reason about.

Weakness:

- Slower than Newton-Raphson.

Professional workflow:

Use bisection as a reliable fallback or baseline. Use Newton when you want
speed and have good initial guesses.

## 5. No-Arbitrage Bounds

Before solving, the price must be economically possible.

For a European call:

```text
max(S - K exp(-rT), 0) <= C <= S
```

For a European put:

```text
max(K exp(-rT) - S, 0) <= P <= K exp(-rT)
```

If a price violates these bounds, an implied volatility may not exist. The
correct response is to reject the input, not force a fake volatility.

## 6. Volatility Smile

If Black-Scholes were a perfect model, all strikes for the same maturity would
have the same implied volatility.

In real markets, implied volatility varies by strike:

```text
strike -> implied volatility
```

This pattern is called the volatility smile or skew.

Why it matters:

- It reveals how the market prices tail risk.
- Equity index options often show higher IV for downside strikes.
- OTM puts can be expensive because investors demand crash protection.

## 7. Code Entry Point

The implementation lives in:

```text
analytics/implied_volatility.py
```

Basic usage:

```python
from analytics.implied_volatility import ImpliedVolatilitySolver
from models.black_scholes import BlackScholesModel
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment

option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
market = MarketEnvironment(spot=100.0, risk_free_rate=0.05, volatility=0.2)

market_price = 13.8579
solver = ImpliedVolatilitySolver(
    option=option,
    market=market,
    market_price=market_price,
)

newton_result = solver.solve_newton(initial_volatility=0.2)
bisection_result = solver.solve_bisection()

print(newton_result.implied_volatility)
print(bisection_result.implied_volatility)
```

## 8. Smile Extraction From Option Chains

For simple option-chain data:

```python
import pandas as pd

from analytics.implied_volatility import extract_implied_volatilities

option_chain = pd.DataFrame(
    {
        "strike": [90.0, 100.0, 110.0],
        "maturity": [1.0, 1.0, 1.0],
        "option_type": ["call", "call", "call"],
        "market_price": [18.14, 10.45, 6.04],
    }
)

smile = extract_implied_volatilities(
    option_chain=option_chain,
    spot=100.0,
    risk_free_rate=0.05,
    method="bisection",
)
```

The output adds:

```text
implied_volatility
moneyness = strike / spot
iv_converged
iv_iterations
```

## 9. Practical Interpretation

Implied volatility is not a forecast in a pure statistical sense. It is the
volatility level embedded in market prices.

When traders say "vol is rich" or "vol is cheap", they are comparing implied
volatility against:

- Historical realized volatility
- Forecast volatility
- Similar strikes and maturities
- Other assets
- Supply and demand for optionality

This is the bridge from theoretical Black-Scholes pricing into real volatility
trading.


# Step 3: Black-Scholes Model

Black-Scholes is the baseline model for European vanilla option pricing. It is
not the final word on real markets, but it is the language that most option
analytics are built around.

## 1. What We Price

European call payoff:

```text
max(S_T - K, 0)
```

European put payoff:

```text
max(K - S_T, 0)
```

Where:

- ``S_T`` is the stock price at maturity.
- ``K`` is the strike.
- The option can only be exercised at maturity.

## 2. Model Inputs

The first Black-Scholes implementation uses:

- ``S``: spot price today.
- ``K``: strike price.
- ``T``: time to maturity in years.
- ``r``: continuously compounded risk-free rate.
- ``sigma``: annualized volatility.

No dividends are included yet. Dividend yield can be added later by replacing
spot terms with dividend-discounted spot terms.

## 3. The Formula

Call price:

```text
C = S N(d1) - K exp(-rT) N(d2)
```

Put price:

```text
P = K exp(-rT) N(-d2) - S N(-d1)
```

Where:

```text
d1 = [ln(S / K) + (r + 0.5 sigma^2)T] / [sigma sqrt(T)]
d2 = d1 - sigma sqrt(T)
```

And ``N(x)`` is the standard normal cumulative distribution function.

## 4. Why d1 And d2 Matter

``d1`` is a volatility-adjusted moneyness term. It combines:

- How far spot is from strike.
- How much time remains.
- How much volatility can move the stock.
- The risk-free growth rate.

``N(d1)`` is closely related to call delta. It tells us how sensitive the call
price is to spot.

``d2`` shifts ``d1`` down by ``sigma sqrt(T)``. ``N(d2)`` appears beside the
discounted strike. It is often described as a risk-neutral probability-like
term that the option finishes in the money, though that description should be
used carefully.

## 5. Pricing Interpretation

The call formula has two economic pieces:

```text
S N(d1)
```

This is the stock-linked benefit of owning the call.

```text
K exp(-rT) N(d2)
```

This is the present value of the strike payment, weighted by the risk-neutral
exercise term.

The put formula mirrors the same idea from the downside perspective.

## 6. Example

Inputs:

```text
S = 100
K = 100
T = 1
r = 0.05
sigma = 0.20
```

Results:

```text
d1 = 0.35
d2 = 0.15
Call price = 10.4506
Put price  = 5.5735
```

These are standard benchmark values and are used in the automated tests.

## 7. Put-Call Parity

For European options with the same strike and maturity:

```text
C - P = S - K exp(-rT)
```

This is a no-arbitrage relationship. If it fails badly in production, either:

- One of the prices is stale.
- The inputs are inconsistent.
- Dividends, funding, or market frictions are missing.
- There is a bug.

## 8. Code Entry Point

The implementation lives in:

```text
models/black_scholes.py
```

Basic usage:

```python
from models.black_scholes import BlackScholesModel
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment

option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
market = MarketEnvironment(spot=100.0, risk_free_rate=0.05, volatility=0.2)

model = BlackScholesModel(option=option, market=market)

print(model.d1)
print(model.d2)
print(model.price())
```

## 9. Validation Checks

The implementation checks:

- Strike must be positive.
- Maturity must be positive.
- Spot must be positive.
- Volatility must be non-negative.
- ``d1`` and ``d2`` cannot be requested when volatility is exactly zero.

When volatility is zero, the stock path is deterministic under the model, so
the price is the discounted deterministic payoff.


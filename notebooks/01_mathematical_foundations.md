# Step 2: Mathematical Foundations

This note explains the stochastic ideas behind the pricing engine we will build.
The objective is not to memorize formulas. The objective is to understand what
each term means economically.

## 1. Randomness In Finance

Option pricing starts from a simple observation: future prices are uncertain.
We need a model that can describe many possible future paths, not just one
forecast.

In a derivatives desk, the model is not trying to predict the exact future
stock price. It is trying to assign consistent values to uncertain payoffs.

## 2. Brownian Motion

Brownian motion, usually written as ``W_t``, is the basic source of continuous
randomness in Black-Scholes.

It has three key properties:

- ``W_0 = 0``.
- Increments are normally distributed: ``W_t - W_s ~ N(0, t - s)``.
- Non-overlapping increments are independent.

Discrete simulation idea:

```text
Delta W = sqrt(Delta t) * Z
Z ~ N(0, 1)
```

Why ``sqrt(Delta t)`` matters:

Variance grows linearly with time, while standard deviation grows with the
square root of time. This is why annual volatility is scaled by ``sqrt(time)``.

Finance intuition:

Brownian motion is the random shock. It is not the stock price itself. It is the
noise that pushes prices around.

## 3. Stochastic Processes

A stochastic process is a variable that evolves through time with randomness.

Examples:

- Stock price: ``S_t``
- Interest rate: ``r_t``
- Volatility: ``sigma_t``
- Brownian shock: ``W_t``

In this project, a path is one possible future history. A simulation creates
many paths so we can estimate expected payoffs.

## 4. Drift Vs Diffusion

A common stochastic differential equation has the form:

```text
dX_t = drift dt + diffusion dW_t
```

Drift:

- The predictable trend component.
- Scales with ``dt``.
- Matters more over longer horizons.

Diffusion:

- The random shock component.
- Scales with ``sqrt(dt)``.
- Dominates over very short horizons.

Trading intuition:

Short-dated option risk is often driven more by diffusion and volatility than
by your view on average return.

## 5. Geometric Brownian Motion

Black-Scholes assumes the stock follows geometric Brownian motion:

```text
dS_t = mu S_t dt + sigma S_t dW_t
```

Where:

- ``S_t`` is the stock price at time ``t``.
- ``mu`` is the real-world expected return.
- ``sigma`` is volatility.
- ``dW_t`` is the random Brownian shock.

The exact solution is:

```text
S_t = S_0 exp((mu - 0.5 sigma^2)t + sigma W_t)
```

Why this model is useful:

- Prices stay positive.
- Log returns are normally distributed.
- Prices are lognormally distributed.
- It leads to closed-form Black-Scholes prices.

Important limitation:

Real markets have jumps, volatility clustering, skew, fat tails, and changing
volatility. That is why later steps add smiles, surfaces, SABR, hedging, and
stress testing.

## 6. Log Returns

Simple return:

```text
R_t = (S_t - S_{t-1}) / S_{t-1}
```

Log return:

```text
r_t = log(S_t / S_{t-1})
```

Why quants like log returns:

- They add across time.
- They are natural under GBM.
- They connect directly to continuously compounded returns.

If a stock moves from 100 to 105 and then 105 to 110, total log return is:

```text
log(105 / 100) + log(110 / 105) = log(110 / 100)
```

## 7. Volatility

Volatility measures the size of random price movements, not their direction.

In GBM:

```text
sigma = annualized standard deviation of log returns
```

If daily log-return volatility is estimated as ``sigma_daily``, annualized
volatility is approximately:

```text
sigma_annual = sigma_daily * sqrt(252)
```

Why volatility matters for options:

Options are convex payoffs. Bigger moves increase the value of optionality,
especially for long calls and puts. This is why option prices rise when
implied volatility rises.

## 8. Risk-Neutral Pricing

Risk-neutral pricing is the central idea behind Black-Scholes.

Under the real-world measure, a stock may have expected return ``mu``:

```text
dS_t = mu S_t dt + sigma S_t dW_t
```

Under the risk-neutral measure, the drift becomes the risk-free rate ``r``:

```text
dS_t = r S_t dt + sigma S_t dW_t^Q
```

The option price is the discounted expected payoff:

```text
Option Price = exp(-rT) * E_Q[Payoff(S_T)]
```

Why this is powerful:

We do not need to know the stock's true expected return to price a derivative
in the Black-Scholes framework. We need the current spot, strike, maturity,
risk-free rate, and volatility.

Trading intuition:

Risk-neutral pricing is not saying investors are actually risk-neutral. It is a
pricing transformation that makes discounted tradable assets behave consistently
with no-arbitrage.

## 9. Simple Visual Mental Model

Brownian motion:

```text
shock path:    0 -> up -> down -> down -> up
meaning:       random accumulated noise
```

GBM stock path:

```text
stock path:    100 -> 102 -> 99 -> 101 -> 108
meaning:       positive price process driven by drift and shocks
```

Option payoff:

```text
call payoff at maturity = max(S_T - K, 0)
put payoff at maturity  = max(K - S_T, 0)
```

Option pricing links the random future distribution of ``S_T`` to today's
fair value.

## 10. Code Entry Points

The reusable Step 2 code lives in:

```text
simulations/stochastic_processes.py
```

It provides:

- ``time_grid``
- ``brownian_increments``
- ``brownian_motion_paths``
- ``geometric_brownian_motion_paths``
- ``log_returns``

These functions will be reused in Step 5 when we build the Monte Carlo engine.


# Step 5: Monte Carlo Simulation Engine

Monte Carlo pricing estimates an option value by simulating many possible
future stock paths and averaging the discounted payoff.

Black-Scholes gives a closed-form formula for European options. Monte Carlo is
more general. It becomes essential when payoffs are path-dependent, portfolios
are complex, or closed-form formulas do not exist.

## 1. Risk-Neutral Simulation

Under the no-dividend Black-Scholes setup, the risk-neutral stock process is:

```text
dS_t = r S_t dt + sigma S_t dW_t
```

The exact GBM simulation formula is:

```text
S_t = S_0 exp((r - 0.5 sigma^2)t + sigma W_t)
```

The important point: in pricing, the drift is ``r``, not a personal forecast
of the stock's expected return.

## 2. Monte Carlo Pricing Formula

For a European option:

```text
Price = exp(-rT) * average(payoff(S_T))
```

Call payoff:

```text
max(S_T - K, 0)
```

Put payoff:

```text
max(K - S_T, 0)
```

## 3. Algorithm

The engine follows this workflow:

```text
1. Simulate many risk-neutral GBM paths.
2. Extract terminal prices S_T.
3. Compute option payoffs at maturity.
4. Discount payoffs back to today.
5. Average discounted payoffs.
6. Report standard error and confidence interval.
```

## 4. Why Vectorization Matters

Python loops are slow for large simulations. NumPy lets us simulate path
matrices directly:

```text
shape = (number of paths, number of time steps + 1)
```

This is closer to professional quant research style: write the math in arrays,
then run large experiments quickly.

## 5. Convergence

Monte Carlo is random. Its error decreases slowly:

```text
standard error ~ 1 / sqrt(number of paths)
```

That means using 4 times more paths only roughly halves the error.

This is why Monte Carlo engines report a standard error. A price without an
error estimate is incomplete.

## 6. Simulation Error

If the analytical Black-Scholes call price is ``10.4506`` and Monte Carlo gives
``10.47`` with standard error ``0.05``, that is a good result.

If Monte Carlo gives ``11.50`` with tiny standard error, something is wrong:

- Wrong drift
- Missing discounting
- Incorrect payoff
- Bug in path generation
- Wrong volatility scaling

## 7. Variance Reduction Intuition

Monte Carlo can be made more efficient by reducing estimator noise.

Common techniques:

- Antithetic variates
- Control variates
- Moment matching
- Importance sampling

We start with plain Monte Carlo first because the clean baseline must be
correct before optimization.

## 8. Code Entry Point

The implementation lives in:

```text
simulations/monte_carlo.py
```

Basic usage:

```python
from models.black_scholes import BlackScholesModel
from simulations.monte_carlo import MonteCarloEngine
from utils.instruments import EuropeanOption, OptionType
from utils.market import MarketEnvironment

option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
market = MarketEnvironment(spot=100.0, risk_free_rate=0.05, volatility=0.2)

mc = MonteCarloEngine(option=option, market=market, n_paths=50_000, steps=252, seed=7)
result = mc.price()

bs_price = BlackScholesModel(option=option, market=market).price()

print(result.price)
print(result.standard_error)
print(result.confidence_interval)
print(bs_price)
```

## 9. Visual Diagnostics

The visualization module includes:

```text
plot_simulated_paths
plot_payoff_distribution
```

These help you inspect whether the simulated paths and terminal payoffs look
reasonable before trusting the numerical result.

## 10. Professional Interpretation

Monte Carlo is not just a pricing trick. It is a framework for:

- Exotic option pricing
- Portfolio P&L simulation
- Hedging error analysis
- Stress testing
- Risk aggregation

That is why we build it early. Later modules will reuse this simulation logic
for dynamic hedging and stress scenarios.


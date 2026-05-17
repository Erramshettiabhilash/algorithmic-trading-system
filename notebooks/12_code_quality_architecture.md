# Step 13: Code Quality & Architecture

Step 13 is about making the platform maintainable.

Quant code is risky when it is clever but unstructured. A professional
derivatives toolkit needs clear module boundaries, validation, tests, and
repeatable quality checks.

## 1. Architecture

The project separates responsibilities:

```text
models/         pricing models
analytics/      Greeks, IV, surfaces, stress, reporting
simulations/    Monte Carlo and hedging paths
visualization/  charts and research outputs
utils/          shared contracts, market data, validation
tests/          numerical and architecture tests
```

This matters because pricing logic should be usable without importing plotting
code, and analytics should be testable without opening notebooks.

## 2. Public API

Package `__init__.py` files expose the stable objects a user is expected to
import.

Example:

```python
from models import BlackScholesModel, SABRModel
from analytics import GreeksEngine, ImpliedVolatilitySolver, StressTester
from simulations import MonteCarloEngine, HedgingSimulator
```

The architecture tests check that exported names actually exist.

## 3. Type Hints And Docstrings

Public functions use type hints and docstrings because quant code has many
similar numeric inputs:

```text
spot, strike, maturity, rate, volatility
```

Without clear signatures, it is too easy to swap arguments or misread units.

## 4. Validation

Inputs are validated near boundaries:

- Spot must be positive.
- Strike must be positive.
- Maturity must be positive.
- Volatility must be non-negative.
- Implied-vol prices must respect no-arbitrage bounds.
- SABR parameters must respect domain constraints.

Validation turns silent numerical nonsense into explicit errors.

## 5. Testing Strategy

The tests cover:

- Known Black-Scholes benchmark prices
- Greeks benchmark values and finite differences
- Monte Carlo convergence against Black-Scholes
- Implied-volatility round trips
- Volatility-surface slices and grids
- SABR calibration recovery
- Hedging accounting and risk neutralization
- Market-making inventory behavior
- Stress-test tail metrics
- Visualization smoke tests
- Public API and architecture checks

## 6. Quality Commands

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

These commands should pass before moving to the final GitHub-ready step.

## 7. Why This Matters In Interviews

A good quant project is not only about formulas. Interviewers also look for:

- Clean decomposition
- Numerical care
- Defensive validation
- Testable design
- Reproducible examples
- Clear explanation of tradeoffs

That is what Step 13 is designed to demonstrate.


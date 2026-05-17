# Architecture

This project is organized like a small derivatives research library. The main
rule is separation of responsibility: pricing models, analytics, simulations,
visualization, and shared domain objects live in different modules.

## Layers

```text
utils/
  Shared financial contracts, market inputs, and validation helpers.

models/
  Mathematical valuation models such as Black-Scholes and SABR.

analytics/
  Risk and research layers built on top of models: Greeks, implied volatility,
  surfaces, stress testing, market making, and reporting.

simulations/
  Path-based engines for Monte Carlo and dynamic hedging.

visualization/
  Plotting utilities that consume arrays, result objects, or DataFrames.

tests/
  Numerical, behavioral, and architecture tests.
```

## Design Principles

- Domain objects are explicit dataclasses.
- Public functions and classes have docstrings and type hints.
- Pricing logic does not depend on plotting code.
- Plotting code accepts already-computed analytics where practical.
- Financial libraries such as QuantLib are intentionally avoided.
- Numerical methods report diagnostics such as convergence, errors, or standard
  error where useful.
- Tests cover known values, no-arbitrage identities, validation failures, and
  stochastic reproducibility.

## Current Public API

The package-level `__init__.py` files export the main user-facing objects:

- `models`: Black-Scholes, SABR, SABR calibration.
- `analytics`: Greeks, IV, surfaces, market making, stress testing, reporting.
- `simulations`: Monte Carlo and hedging.
- `visualization`: chart helpers.
- `utils`: option and market dataclasses plus validation.

## Extension Pattern

When adding a new module:

1. Put mathematical valuation logic in `models/`.
2. Put risk or workflow analytics in `analytics/`.
3. Put path-dependent experiments in `simulations/`.
4. Put charts in `visualization/`.
5. Add focused unit tests.
6. Export stable public objects from the package `__init__.py` only when they
   are meant for users.


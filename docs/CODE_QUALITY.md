# Code Quality Checklist

Use this checklist before committing or presenting the project.

## Local Checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

## Review Checklist

- Public functions and classes have clear docstrings.
- Function signatures use type hints.
- Inputs are validated at module boundaries.
- Numerical methods expose diagnostics such as convergence, error, standard
  error, RMSE, VaR, or expected shortfall.
- Tests include both happy paths and failure paths.
- Stochastic tests use fixed seeds.
- Plotting tests save files instead of requiring a GUI backend.
- Generated outputs stay in `results/`.
- Large raw datasets stay out of Git.

## Numerical Stability Practices

- Validate positivity constraints for spot, strike, maturity, and volatility.
- Handle zero-volatility edge cases explicitly.
- Use robust methods such as bisection when Newton-style solvers may fail.
- Add special ATM formulas when a model has a removable singularity, as in SABR.
- Report Monte Carlo standard error and confidence intervals.
- Treat stress-test tail metrics as diagnostics, not precise forecasts.


# AI Quant Platform Code Quality Guide

This guide describes the engineering standards used by the AI-driven quant research platform.

## Local Quality Gates

Run before presenting or committing:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

## Architecture Rules

- Keep data ingestion in `data/`.
- Keep feature engineering in `features/`.
- Keep predictive models in `models/`.
- Keep model evaluation in `evaluation/`.
- Keep trading execution simulation in `trading/`.
- Keep portfolio and factor risk analytics in `risk/`.
- Keep reports and visuals in `analytics/` and `visualization/`.

Each module should have one reason to change.

## Public API Rules

- Every public package exports a curated `__all__`.
- Public modules have module docstrings.
- Public functions use type hints.
- Packages include `py.typed` markers for downstream type-aware users.
- Heavy optional dependencies are imported lazily when practical.

## Numerical Validation Rules

Quant bugs are often silent. Validate:

- required columns
- timestamp indexes
- aligned indexes
- finite values
- positive volatility, capital, windows, and sizes
- no empty frames after joins or cleaning

Use shared helpers from `utils.validation` where possible.

## Testing Rules

- Unit tests should use deterministic synthetic data.
- Stochastic code should use fixed seeds.
- Time-series tests should verify chronological order.
- Plotting tests should save files using a non-GUI backend.
- Optimization tests should use small trial counts.
- Deep learning and RL tests should be smoke tests, not long research runs.

## Finance-Specific Review Checklist

- Are targets aligned to the decision timestamp?
- Are validation splits chronological?
- Are costs and turnover included in trading simulations?
- Are factor exposures visible?
- Are model explanations available?
- Are walk-forward results out-of-sample?
- Are reports reproducible from code?

## Interview-Ready Explanation

The platform is engineered as modular research infrastructure. Data, features, models, evaluation,
trading, risk, explainability, and visualization are separated so experiments are reproducible and
mistakes are easier to isolate. The codebase uses type hints, docstrings, tests, curated public
exports, and numerical validation guards to reduce the chance of silent financial ML errors.

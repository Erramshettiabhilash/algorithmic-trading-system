# Step 1 - Project Setup and Architecture

## Goal

Build the skeleton for a machine-learning-driven quant research platform. At this stage we are
not trying to build the full trading engine yet. We are defining the boundaries that keep future
research clean, testable, and reproducible.

## Required Tools

- Python: the main language because the quant and ML ecosystem is strongest here.
- Pandas: time-series dataframes, OHLCV cleaning, joins, resampling, and rolling features.
- NumPy: vectorized numerical operations behind most return and risk calculations.
- Scikit-learn: baseline ML models, preprocessing, validation patterns, and metrics.
- XGBoost: strong tabular nonlinear model for factor interactions and noisy finance data.
- PyTorch: deep learning framework for LSTM and reinforcement learning modules.
- SHAP: explains feature contribution so predictions can be audited.
- Optuna: Bayesian hyperparameter optimization without brute-force grid search.
- Matplotlib: reliable static charts for reports and tests.
- Plotly: interactive charts for dashboards and research review.

## Professional Pipeline

The platform follows the same broad loop used in institutional research:

1. Load and clean market data.
2. Generate timestamp-safe predictive features.
3. Define forward-looking targets without leakage.
4. Train models only on past data.
5. Evaluate both prediction quality and trading quality.
6. Explain model behavior.
7. Convert predictions into signals and portfolio decisions.
8. Run walk-forward validation and produce research reports.

## Step 1 Checkpoint

After this step, the repository has the folders, dependencies, starter interfaces, and architecture
documentation needed to begin Step 2: data collection and preprocessing.

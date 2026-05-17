"""Run an end-to-end AI quant research demo and save example outputs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import generate_factor_analytics_report, generate_model_evaluation_report
from data import clean_ohlcv
from evaluation import (
    WalkForwardConfig,
    build_supervised_dataset,
    create_forward_return_target,
    evaluate_factor_predictions,
    run_walk_forward_research,
)
from explainability import ShapExplainabilityEngine
from features import FeatureConfig, QuantFeatureEngine
from models import XGBoostFactorModel, XGBoostModelConfig
from regime import RegimeDetectionConfig, classify_market_regimes
from risk import FactorRiskModel, compute_risk_contribution
from trading import (
    ExecutionConfig,
    PositionSizingConfig,
    SignalConfig,
    construct_long_short_weights,
    simulate_multi_asset_portfolio,
    simulate_signal_strategy,
)
from visualization import (
    plot_drawdown_curve,
    plot_equity_curve,
    plot_performance_dashboard,
    plot_portfolio_allocation,
    plot_prediction_vs_actual,
    plot_regime_classification,
    plot_shap_importance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "examples" / "ai_quant_demo"


def main() -> None:
    """Generate a compact AI quant research showcase."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = _synthetic_ohlcv()
    prices.to_csv(OUTPUT_DIR / "synthetic_ohlcv.csv")

    feature_config = FeatureConfig(
        return_windows=(1, 5, 10),
        volatility_windows=(10, 20),
        momentum_windows=(10, 20),
        rsi_window=10,
        ema_fast=8,
        ema_slow=18,
        macd_signal=6,
        atr_window=10,
        volume_window=10,
        structure_window=15,
    )
    clean_prices = clean_ohlcv(prices)
    features = QuantFeatureEngine(feature_config).transform(clean_prices)
    target = create_forward_return_target(clean_prices, horizon=1)
    x, y = build_supervised_dataset(features, target)

    model_config = XGBoostModelConfig(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.05,
        n_jobs=1,
    )
    split_index = int(len(x) * 0.75)
    x_train, x_test = x.iloc[:split_index], x.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = XGBoostFactorModel(model_config).fit(x_train, y_train)
    predictions = model.predict(x_test).predictions
    evaluation = evaluate_factor_predictions(predictions, y_test)

    backtest = simulate_signal_strategy(
        predictions,
        y_test,
        signal_config=SignalConfig(
            long_threshold=-0.002,
            short_threshold=-0.002,
            allow_short=False,
        ),
        sizing_config=PositionSizingConfig(max_position=1.0),
        execution_config=ExecutionConfig(transaction_cost_bps=1.0, slippage_bps=0.5),
    )

    walk_forward = run_walk_forward_research(
        x,
        y,
        model_factory=lambda: XGBoostFactorModel(model_config),
        config=WalkForwardConfig(
            mode="expanding",
            initial_train_size=80,
            test_size=20,
            step_size=20,
        ),
    )
    walk_forward.fold_summary.to_csv(OUTPUT_DIR / "walk_forward_summary.csv")

    regimes = classify_market_regimes(
        clean_prices,
        RegimeDetectionConfig(return_window=10, volatility_window=10),
    )

    shap_engine = ShapExplainabilityEngine(model)
    shap_explanation = shap_engine.explain(x_test.tail(40))
    shap_importance = shap_engine.feature_importance(shap_explanation)
    shap_importance.to_csv(OUTPUT_DIR / "shap_importance.csv")

    asset_returns = _synthetic_asset_returns(y)
    factor_returns = _synthetic_factor_returns(asset_returns.index)
    weights = pd.Series({"asset_a": 0.45, "asset_b": 0.35, "asset_c": 0.20})
    risk_model = FactorRiskModel().fit(asset_returns, factor_returns)
    exposure = risk_model.portfolio_exposure(weights)
    risk_contribution = compute_risk_contribution(weights, asset_returns.cov() * 252)

    multi_asset_predictions = _synthetic_multi_asset_predictions(asset_returns.index)
    allocation = construct_long_short_weights(
        multi_asset_predictions,
        gross_leverage=1.0,
        top_n=1,
    )
    portfolio_backtest = simulate_multi_asset_portfolio(
        allocation,
        asset_returns,
        execution_config=ExecutionConfig(transaction_cost_bps=1.0),
    )

    plot_prediction_vs_actual(
        predictions,
        y_test,
        OUTPUT_DIR / "prediction_vs_actual.png",
    )
    plot_equity_curve(backtest.strategy_returns, OUTPUT_DIR / "equity_curve.png")
    plot_drawdown_curve(backtest.strategy_returns, OUTPUT_DIR / "drawdown_curve.png")
    plot_performance_dashboard(
        backtest.strategy_returns,
        OUTPUT_DIR / "performance_dashboard.png",
    )
    plot_shap_importance(shap_importance, OUTPUT_DIR / "shap_importance.png")
    plot_regime_classification(
        clean_prices["close"],
        regimes,
        OUTPUT_DIR / "regime_classification.png",
    )
    plot_portfolio_allocation(allocation, OUTPUT_DIR / "portfolio_allocation.html")

    generate_model_evaluation_report(
        OUTPUT_DIR / "model_evaluation_report.md",
        title="AI Quant XGBoost Walk-Forward Demo",
        metrics=backtest.performance,
        notes=[
            "Synthetic data is used for reproducibility.",
            "Signals include transaction costs and slippage.",
            "Use real data only after validating timestamp availability.",
        ],
    )
    generate_factor_analytics_report(
        OUTPUT_DIR / "factor_analytics_report.md",
        exposure=exposure.portfolio_exposure,
        risk_contribution=risk_contribution.percent_contribution,
    )

    summary = {
        "model_evaluation": asdict(evaluation),
        "trading_performance": asdict(backtest.performance),
        "walk_forward_evaluation": asdict(walk_forward.evaluation),
        "portfolio_performance": asdict(portfolio_backtest.performance),
        "top_shap_features": shap_importance.head(5).to_dict(),
        "portfolio_factor_exposure": exposure.portfolio_exposure.to_dict(),
    }
    (OUTPUT_DIR / "ai_quant_demo_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote AI quant demo outputs to {OUTPUT_DIR}")


def _synthetic_ohlcv(rows: int = 220) -> pd.DataFrame:
    """Create deterministic OHLCV data with trend, cycles, and volatility shifts."""

    rng = np.random.default_rng(42)
    index = pd.date_range("2023-01-02", periods=rows, freq="B", tz="UTC")
    trend = np.linspace(100.0, 125.0, rows)
    cycle = 3.0 * np.sin(np.linspace(0.0, 14.0, rows))
    noise = rng.normal(0.0, 0.35, rows).cumsum()
    close = trend + cycle + noise
    open_ = close * (1.0 + 0.002 * np.cos(np.linspace(0.0, 8.0, rows)))
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8
    volume = 1_000_000 + 75_000 * np.sin(np.linspace(0.0, 12.0, rows))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def _synthetic_asset_returns(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create deterministic asset returns for portfolio and risk demos."""

    base = pd.Series(np.sin(np.linspace(0.0, 10.0, len(index))) * 0.004, index=index)
    return pd.DataFrame(
        {
            "asset_a": base + 0.002,
            "asset_b": -0.5 * base + 0.001,
            "asset_c": 0.3 * base - 0.0005,
        },
        index=index,
    )


def _synthetic_factor_returns(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create deterministic factor returns for risk analytics."""

    market = pd.Series(np.sin(np.linspace(0.0, 8.0, len(index))) * 0.003, index=index)
    momentum = pd.Series(np.cos(np.linspace(0.0, 6.0, len(index))) * 0.002, index=index)
    volatility = pd.Series(np.sin(np.linspace(0.0, 5.0, len(index))) * 0.0015, index=index)
    return pd.DataFrame(
        {"market": market, "momentum": momentum, "volatility": volatility},
        index=index,
    )


def _synthetic_multi_asset_predictions(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create deterministic cross-sectional predictions for allocation demos."""

    return pd.DataFrame(
        {
            "asset_a": np.sin(np.linspace(0.0, 6.0, len(index))),
            "asset_b": np.cos(np.linspace(0.0, 6.0, len(index))),
            "asset_c": -np.sin(np.linspace(0.0, 4.0, len(index))),
        },
        index=index,
    )


if __name__ == "__main__":
    main()

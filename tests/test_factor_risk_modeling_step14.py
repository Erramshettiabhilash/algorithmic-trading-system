"""Tests for Step 14 factor risk modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk import (
    FactorRiskModel,
    compute_factor_covariance,
    compute_risk_contribution,
    create_volatility_factor,
    estimate_factor_betas,
)


def _factor_data(rows: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2024-01-01", periods=rows, freq="B", tz="UTC")
    market = np.linspace(-0.02, 0.025, rows)
    momentum = 0.01 * np.sin(np.linspace(0.0, 10.0, rows))
    volatility = 0.008 * np.cos(np.linspace(0.0, 8.0, rows))
    factors = pd.DataFrame(
        {
            "market": market,
            "momentum": momentum,
            "volatility": volatility,
        },
        index=index,
    )
    assets = pd.DataFrame(
        {
            "asset_a": 0.001 + 1.2 * market + 0.5 * momentum - 0.2 * volatility,
            "asset_b": -0.0005 + 0.7 * market - 0.3 * momentum + 0.8 * volatility,
            "asset_c": 0.0002 - 0.2 * market + 0.9 * momentum + 0.1 * volatility,
        },
        index=index,
    )
    return assets, factors


def test_estimate_factor_betas_recovers_known_exposures() -> None:
    assets, factors = _factor_data()
    result = estimate_factor_betas(assets, factors)

    assert abs(result.betas.loc["asset_a", "market"] - 1.2) < 1e-8
    assert abs(result.betas.loc["asset_b", "volatility"] - 0.8) < 1e-8
    assert result.r_squared.min() > 0.99
    assert result.residual_volatility.max() < 1e-10


def test_factor_risk_model_portfolio_exposure() -> None:
    assets, factors = _factor_data()
    model = FactorRiskModel().fit(assets, factors)
    weights = pd.Series({"asset_a": 0.5, "asset_b": 0.3, "asset_c": 0.2})

    report = model.portfolio_exposure(weights)

    expected_market = 0.5 * 1.2 + 0.3 * 0.7 + 0.2 * -0.2
    assert abs(report.portfolio_exposure["market"] - expected_market) < 1e-8
    assert report.gross_exposure == 1.0
    assert report.net_exposure == 1.0


def test_factor_decomposition_reconciles_realized_returns() -> None:
    assets, factors = _factor_data()
    model = FactorRiskModel().fit(assets, factors)
    weights = pd.Series({"asset_a": 0.5, "asset_b": 0.3, "asset_c": 0.2})
    decomposition = model.decompose_returns(weights, factors, assets)

    reconstructed = decomposition.factor_return + decomposition.residual_return

    assert np.allclose(reconstructed, decomposition.realized_return)
    assert set(decomposition.factor_contribution.columns) == set(factors.columns)


def test_factor_covariance_and_risk_contribution() -> None:
    assets, factors = _factor_data()
    covariance = compute_factor_covariance(factors)
    asset_covariance = assets.cov() * 252
    weights = pd.Series({"asset_a": 0.5, "asset_b": 0.3, "asset_c": 0.2})
    contribution = compute_risk_contribution(weights, asset_covariance)

    assert covariance.shape == (3, 3)
    assert contribution.portfolio_volatility > 0.0
    assert abs(contribution.percent_contribution.sum() - 1.0) < 1e-10


def test_create_volatility_factor() -> None:
    assets, _ = _factor_data()
    factor = create_volatility_factor(assets["asset_a"], window=10)

    assert factor.name == "realized_volatility_10"
    assert factor.dropna().ge(0.0).all()

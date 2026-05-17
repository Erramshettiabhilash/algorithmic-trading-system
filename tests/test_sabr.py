"""Tests for the Step 8 SABR stochastic-volatility model."""

from __future__ import annotations

import numpy as np
import pytest

from models.sabr import SABRModel, calibrate_sabr_smile
from visualization.charts import plot_sabr_fit


def test_sabr_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="alpha must be positive"):
        SABRModel(alpha=0.0, beta=0.5, rho=0.0, nu=0.5)

    with pytest.raises(ValueError, match="beta must be less"):
        SABRModel(alpha=0.2, beta=1.5, rho=0.0, nu=0.5)

    with pytest.raises(ValueError, match="rho must be strictly"):
        SABRModel(alpha=0.2, beta=0.5, rho=1.0, nu=0.5)


def test_sabr_atm_implied_volatility_is_positive() -> None:
    model = SABRModel(alpha=2.0, beta=0.5, rho=-0.3, nu=0.7)

    volatility = model.implied_volatility(forward=100.0, strike=100.0, maturity=1.0)

    assert volatility > 0.0
    assert volatility == pytest.approx(0.2061, abs=1e-4)


def test_sabr_general_formula_is_continuous_near_atm() -> None:
    model = SABRModel(alpha=2.0, beta=0.5, rho=-0.3, nu=0.7)
    atm = model.implied_volatility(forward=100.0, strike=100.0, maturity=1.0)
    near_atm = model.implied_volatility(forward=100.0, strike=100.000001, maturity=1.0)

    assert near_atm == pytest.approx(atm, abs=1e-6)


def test_sabr_smile_shape_and_equity_style_skew() -> None:
    model = SABRModel(alpha=2.0, beta=0.5, rho=-0.35, nu=0.8)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])

    smile = model.smile(forward=100.0, strikes=strikes, maturity=1.0)

    assert smile.shape == strikes.shape
    assert np.all(smile > 0.0)
    assert smile[0] > smile[-1]


def test_sabr_calibration_recovers_synthetic_smile() -> None:
    true_model = SABRModel(alpha=2.0, beta=0.5, rho=-0.35, nu=0.8)
    strikes = np.array([70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0])
    market_vols = true_model.smile(forward=100.0, strikes=strikes, maturity=1.0)

    result = calibrate_sabr_smile(
        forward=100.0,
        strikes=strikes,
        market_volatilities=market_vols,
        maturity=1.0,
        beta=0.5,
        initial_guess=(1.5, -0.1, 0.5),
    )

    fitted_vols = result.model.smile(forward=100.0, strikes=strikes, maturity=1.0)

    assert result.success
    assert result.rmse < 1e-8
    np.testing.assert_allclose(fitted_vols, market_vols, atol=1e-8)
    assert result.model.alpha == pytest.approx(2.0, abs=1e-5)
    assert result.model.rho == pytest.approx(-0.35, abs=1e-5)
    assert result.model.nu == pytest.approx(0.8, abs=1e-5)


def test_sabr_calibration_validates_inputs() -> None:
    strikes = np.array([90.0, 100.0])
    market_vols = np.array([0.22, 0.20])

    with pytest.raises(ValueError, match="at least three"):
        calibrate_sabr_smile(
            forward=100.0,
            strikes=strikes,
            market_volatilities=market_vols,
            maturity=1.0,
        )


def test_plot_sabr_fit_returns_matplotlib_figure(tmp_path) -> None:
    strikes = np.array([90.0, 100.0, 110.0])
    market_vols = np.array([0.24, 0.20, 0.21])
    sabr_vols = np.array([0.235, 0.205, 0.208])

    fig = plot_sabr_fit(strikes, market_vols, sabr_vols, output_path=str(tmp_path / "sabr.png"))

    assert len(fig.axes) == 1

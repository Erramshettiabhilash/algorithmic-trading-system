"""SABR stochastic-volatility model and smile calibration.

This module implements the Hagan et al. lognormal implied-volatility
approximation. SABR is commonly used to fit and interpolate volatility smiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from utils.validation import require_non_negative, require_positive

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SABRCalibrationResult:
    """Result object returned by SABR smile calibration."""

    model: SABRModel
    rmse: float
    success: bool
    iterations: int
    objective_value: float


@dataclass(frozen=True)
class SABRModel:
    """SABR model parameters for lognormal implied volatility.

    SABR dynamics:

    ``dF_t = alpha_t F_t^beta dW_t``
    ``d alpha_t = nu alpha_t dZ_t``
    ``corr(dW_t, dZ_t) = rho``

    Args:
        alpha: Initial volatility level.
        beta: Backbone parameter in ``[0, 1]``.
        rho: Correlation between forward and volatility shocks, in ``(-1, 1)``.
        nu: Volatility of volatility.
    """

    alpha: float
    beta: float
    rho: float
    nu: float

    def __post_init__(self) -> None:
        """Validate SABR parameters."""
        require_positive(self.alpha, "alpha")
        require_non_negative(self.beta, "beta")
        require_non_negative(self.nu, "nu")
        if self.beta > 1.0:
            raise ValueError(f"beta must be less than or equal to 1. Received {self.beta}.")
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"rho must be strictly between -1 and 1. Received {self.rho}.")

    def implied_volatility(self, forward: float, strike: float, maturity: float) -> float:
        """Return Hagan lognormal SABR implied volatility."""
        require_positive(forward, "forward")
        require_positive(strike, "strike")
        require_positive(maturity, "maturity")

        if abs(forward - strike) < 1e-12:
            return self._atm_implied_volatility(forward=forward, maturity=maturity)

        one_minus_beta = 1.0 - self.beta
        log_fk = log(forward / strike)
        fk_beta = (forward * strike) ** (0.5 * one_minus_beta)
        denominator_adjustment = (
            1.0
            + (one_minus_beta**2 / 24.0) * log_fk**2
            + (one_minus_beta**4 / 1920.0) * log_fk**4
        )
        denominator = fk_beta * denominator_adjustment

        z = (self.nu / self.alpha) * fk_beta * log_fk
        z_over_xz = self._z_over_x(z)
        time_adjustment = self._time_adjustment(
            forward=forward,
            strike=strike,
            maturity=maturity,
        )

        return (self.alpha / denominator) * z_over_xz * time_adjustment

    def smile(
        self,
        forward: float,
        strikes: FloatArray,
        maturity: float,
    ) -> FloatArray:
        """Return SABR implied volatilities across strikes."""
        if strikes.ndim != 1:
            raise ValueError("strikes must be a one-dimensional array.")

        vols = [
            self.implied_volatility(forward=forward, strike=float(strike), maturity=maturity)
            for strike in strikes
        ]
        return np.array(vols, dtype=np.float64)

    def _atm_implied_volatility(self, forward: float, maturity: float) -> float:
        """Return the ATM Hagan approximation."""
        one_minus_beta = 1.0 - self.beta
        forward_beta = forward**one_minus_beta
        time_adjustment = (
            1.0
            + (
                (one_minus_beta**2 / 24.0) * self.alpha**2 / forward ** (2.0 * one_minus_beta)
                + (self.rho * self.beta * self.nu * self.alpha)
                / (4.0 * forward_beta)
                + ((2.0 - 3.0 * self.rho**2) / 24.0) * self.nu**2
            )
            * maturity
        )
        return (self.alpha / forward_beta) * time_adjustment

    def _time_adjustment(self, forward: float, strike: float, maturity: float) -> float:
        """Return the Hagan maturity adjustment term."""
        one_minus_beta = 1.0 - self.beta
        fk = forward * strike
        fk_beta = fk ** (0.5 * one_minus_beta)
        correction = (
            (one_minus_beta**2 / 24.0) * self.alpha**2 / fk**one_minus_beta
            + (self.rho * self.beta * self.nu * self.alpha) / (4.0 * fk_beta)
            + ((2.0 - 3.0 * self.rho**2) / 24.0) * self.nu**2
        )
        return 1.0 + correction * maturity

    def _z_over_x(self, z: float) -> float:
        """Return ``z / x(z)`` with a stable small-z fallback."""
        if abs(z) < 1e-8 or self.nu == 0.0:
            return 1.0

        discriminant = sqrt(1.0 - 2.0 * self.rho * z + z**2)
        numerator = discriminant + z - self.rho
        denominator = 1.0 - self.rho
        x_z = log(numerator / denominator)

        if abs(x_z) < 1e-12:
            return 1.0
        return z / x_z


def calibrate_sabr_smile(
    forward: float,
    strikes: FloatArray,
    market_volatilities: FloatArray,
    maturity: float,
    beta: float = 0.5,
    initial_guess: tuple[float, float, float] = (0.25, -0.2, 0.5),
    weights: FloatArray | None = None,
) -> SABRCalibrationResult:
    """Calibrate ``alpha``, ``rho``, and ``nu`` to one market smile.

    Args:
        forward: Current forward price for the maturity.
        strikes: Strike vector.
        market_volatilities: Market implied volatilities for the strikes.
        maturity: Option maturity in years.
        beta: Fixed SABR beta parameter.
        initial_guess: Initial ``(alpha, rho, nu)``.
        weights: Optional residual weights.

    Returns:
        Calibration result containing the fitted model and error diagnostics.
    """
    require_positive(forward, "forward")
    require_positive(maturity, "maturity")
    require_non_negative(beta, "beta")
    if beta > 1.0:
        raise ValueError(f"beta must be less than or equal to 1. Received {beta}.")
    _validate_calibration_arrays(strikes, market_volatilities, weights)

    alpha_guess, rho_guess, nu_guess = initial_guess
    initial = np.array(
        [
            max(alpha_guess, 1e-6),
            min(max(rho_guess, -0.999), 0.999),
            max(nu_guess, 0.0),
        ],
        dtype=np.float64,
    )

    residual_weights = (
        np.ones_like(market_volatilities, dtype=np.float64)
        if weights is None
        else weights.astype(np.float64)
    )

    def residuals(parameters: FloatArray) -> FloatArray:
        alpha, rho, nu = parameters
        model = SABRModel(alpha=float(alpha), beta=beta, rho=float(rho), nu=float(nu))
        model_vols = model.smile(forward=forward, strikes=strikes, maturity=maturity)
        return residual_weights * (model_vols - market_volatilities)

    result = least_squares(
        residuals,
        x0=initial,
        bounds=([1e-8, -0.999, 0.0], [5.0, 0.999, 5.0]),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=2_000,
    )

    alpha, rho, nu = result.x
    calibrated_model = SABRModel(alpha=float(alpha), beta=beta, rho=float(rho), nu=float(nu))
    unweighted_errors = (
        calibrated_model.smile(forward=forward, strikes=strikes, maturity=maturity)
        - market_volatilities
    )
    rmse = float(np.sqrt(np.mean(unweighted_errors**2)))

    return SABRCalibrationResult(
        model=calibrated_model,
        rmse=rmse,
        success=bool(result.success),
        iterations=int(result.nfev),
        objective_value=float(result.cost),
    )


def _validate_calibration_arrays(
    strikes: FloatArray,
    market_volatilities: FloatArray,
    weights: FloatArray | None,
) -> None:
    """Validate calibration vector inputs."""
    if strikes.ndim != 1 or market_volatilities.ndim != 1:
        raise ValueError("strikes and market_volatilities must be one-dimensional arrays.")
    if strikes.size != market_volatilities.size:
        raise ValueError("strikes and market_volatilities must have the same length.")
    if strikes.size < 3:
        raise ValueError("at least three strike points are required for SABR calibration.")
    if np.any(strikes <= 0.0):
        raise ValueError("all strikes must be positive.")
    if np.any(market_volatilities < 0.0):
        raise ValueError("all market volatilities must be non-negative.")
    if weights is not None:
        if weights.ndim != 1 or weights.size != strikes.size:
            raise ValueError("weights must be one-dimensional and match strikes.")
        if np.any(weights < 0.0):
            raise ValueError("weights must be non-negative.")
    if not np.all(np.isfinite(strikes)) or not np.all(np.isfinite(market_volatilities)):
        raise ValueError("strikes and market_volatilities must be finite.")
    if weights is not None and not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite.")
    if not all(isfinite(float(value)) for value in strikes):
        raise ValueError("strikes must be finite.")


"""Factor risk analytics for systematic trading portfolios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorRiskResult:
    """Asset-level regression output from a linear factor risk model."""

    betas: pd.DataFrame
    alphas: pd.Series
    residual_returns: pd.DataFrame
    residual_volatility: pd.Series
    r_squared: pd.Series


@dataclass(frozen=True)
class FactorExposureReport:
    """Portfolio exposure to systematic factors."""

    asset_weights: pd.Series
    asset_betas: pd.DataFrame
    portfolio_exposure: pd.Series
    gross_exposure: float
    net_exposure: float


@dataclass(frozen=True)
class FactorDecomposition:
    """Return attribution into factor and residual components."""

    realized_return: pd.Series
    factor_return: pd.Series
    residual_return: pd.Series
    factor_contribution: pd.DataFrame


@dataclass(frozen=True)
class RiskContributionReport:
    """Asset-level contribution to portfolio volatility."""

    weights: pd.Series
    covariance: pd.DataFrame
    marginal_contribution: pd.Series
    component_contribution: pd.Series
    percent_contribution: pd.Series
    portfolio_volatility: float


class FactorRiskModel:
    """Linear factor risk model for asset returns."""

    def __init__(self) -> None:
        self.result: FactorRiskResult | None = None

    def fit(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> FactorRiskModel:
        """Estimate factor betas for each asset."""

        self.result = estimate_factor_betas(asset_returns, factor_returns)
        return self

    def portfolio_exposure(self, weights: pd.Series) -> FactorExposureReport:
        """Compute portfolio factor exposure from fitted asset betas."""

        if self.result is None:
            raise ValueError("factor risk model must be fit before exposure analysis")
        return compute_portfolio_factor_exposure(weights, self.result.betas)

    def decompose_returns(
        self,
        weights: pd.Series,
        factor_returns: pd.DataFrame,
        asset_returns: pd.DataFrame,
    ) -> FactorDecomposition:
        """Decompose portfolio returns into factor and residual return streams."""

        if self.result is None:
            raise ValueError("factor risk model must be fit before return decomposition")
        exposure = self.portfolio_exposure(weights).portfolio_exposure
        return decompose_portfolio_returns(
            weights=weights,
            asset_returns=asset_returns,
            factor_returns=factor_returns,
            portfolio_factor_exposure=exposure,
        )


def estimate_factor_betas(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> FactorRiskResult:
    """Estimate asset betas to systematic factors using ordinary least squares."""

    aligned_assets, aligned_factors = _align_returns(asset_returns, factor_returns)
    x = aligned_factors.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])

    betas: dict[str, pd.Series] = {}
    alphas: dict[str, float] = {}
    residuals: dict[str, pd.Series] = {}
    residual_volatility: dict[str, float] = {}
    r_squared: dict[str, float] = {}

    for asset in aligned_assets.columns:
        y = aligned_assets[asset].to_numpy(dtype=float)
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coefficients
        residual = y - fitted
        total_sum_squares = float(np.square(y - y.mean()).sum())
        residual_sum_squares = float(np.square(residual).sum())
        r2 = (
            0.0
            if total_sum_squares == 0.0
            else 1.0 - residual_sum_squares / total_sum_squares
        )

        alphas[asset] = float(coefficients[0])
        betas[asset] = pd.Series(coefficients[1:], index=aligned_factors.columns)
        residuals[asset] = pd.Series(residual, index=aligned_assets.index)
        residual_volatility[asset] = float(np.std(residual, ddof=1))
        r_squared[asset] = float(r2)

    return FactorRiskResult(
        betas=pd.DataFrame(betas).T,
        alphas=pd.Series(alphas, name="alpha"),
        residual_returns=pd.DataFrame(residuals),
        residual_volatility=pd.Series(residual_volatility, name="residual_volatility"),
        r_squared=pd.Series(r_squared, name="r_squared"),
    )


def compute_portfolio_factor_exposure(
    weights: pd.Series,
    asset_betas: pd.DataFrame,
) -> FactorExposureReport:
    """Compute portfolio-level factor exposure as weighted asset betas."""

    aligned_weights = weights.reindex(asset_betas.index).fillna(0.0).astype(float)
    exposure = asset_betas.mul(aligned_weights, axis=0).sum(axis=0)
    exposure.name = "portfolio_factor_exposure"
    return FactorExposureReport(
        asset_weights=aligned_weights,
        asset_betas=asset_betas,
        portfolio_exposure=exposure,
        gross_exposure=float(aligned_weights.abs().sum()),
        net_exposure=float(aligned_weights.sum()),
    )


def compute_factor_covariance(factor_returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """Compute factor covariance matrix."""

    if factor_returns.empty:
        raise ValueError("factor_returns cannot be empty")

    covariance = factor_returns.dropna().cov()
    if annualize:
        covariance = covariance * 252
    return covariance


def compute_risk_contribution(
    weights: pd.Series,
    asset_covariance: pd.DataFrame,
) -> RiskContributionReport:
    """Compute each asset's contribution to portfolio volatility."""

    aligned_covariance = asset_covariance.reindex(index=weights.index, columns=weights.index)
    if aligned_covariance.isna().any().any():
        raise ValueError("asset covariance matrix must contain all weighted assets")

    weight_values = weights.astype(float).to_numpy()
    covariance_values = aligned_covariance.to_numpy(dtype=float)
    portfolio_variance = float(weight_values.T @ covariance_values @ weight_values)
    if portfolio_variance <= 0.0:
        raise ValueError("portfolio variance must be positive")

    portfolio_volatility = float(np.sqrt(portfolio_variance))
    marginal = covariance_values @ weight_values / portfolio_volatility
    component = weight_values * marginal
    percent = component / portfolio_volatility

    return RiskContributionReport(
        weights=weights,
        covariance=aligned_covariance,
        marginal_contribution=pd.Series(marginal, index=weights.index, name="marginal_risk"),
        component_contribution=pd.Series(component, index=weights.index, name="component_risk"),
        percent_contribution=pd.Series(percent, index=weights.index, name="percent_risk"),
        portfolio_volatility=portfolio_volatility,
    )


def decompose_portfolio_returns(
    *,
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    portfolio_factor_exposure: pd.Series,
) -> FactorDecomposition:
    """Decompose weighted portfolio returns into factor and residual components."""

    aligned_assets, aligned_factors = _align_returns(asset_returns, factor_returns)
    aligned_weights = weights.reindex(aligned_assets.columns).fillna(0.0).astype(float)
    realized = aligned_assets.mul(aligned_weights, axis=1).sum(axis=1)

    exposure = portfolio_factor_exposure.reindex(aligned_factors.columns).fillna(0.0)
    factor_contribution = aligned_factors.mul(exposure, axis=1)
    factor_return = factor_contribution.sum(axis=1)
    residual_return = realized - factor_return

    return FactorDecomposition(
        realized_return=realized.rename("realized_portfolio_return"),
        factor_return=factor_return.rename("factor_return"),
        residual_return=residual_return.rename("residual_return"),
        factor_contribution=factor_contribution,
    )


def create_volatility_factor(
    returns: pd.Series,
    *,
    window: int = 20,
    annualize: bool = True,
) -> pd.Series:
    """Create a realized-volatility factor from return data."""

    if window < 2:
        raise ValueError("window must be at least 2")

    volatility = returns.rolling(window).std(ddof=1)
    if annualize:
        volatility = volatility * np.sqrt(252)
    volatility.name = f"realized_volatility_{window}"
    return volatility


def _align_returns(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align asset and factor returns by timestamp and remove missing rows."""

    if asset_returns.empty:
        raise ValueError("asset_returns cannot be empty")
    if factor_returns.empty:
        raise ValueError("factor_returns cannot be empty")

    aligned = pd.concat(
        {"asset": asset_returns, "factor": factor_returns},
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        raise ValueError("asset and factor returns have no aligned observations")

    return aligned["asset"], aligned["factor"]

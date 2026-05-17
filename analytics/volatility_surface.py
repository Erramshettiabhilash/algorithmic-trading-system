"""Volatility smile, skew, term-structure, and surface analytics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from utils.validation import require_positive

REQUIRED_SURFACE_COLUMNS = {"strike", "maturity", "implied_volatility"}


@dataclass(frozen=True)
class SurfaceGrid:
    """Rectangular volatility surface grid for 3D plotting and diagnostics."""

    strikes: NDArray[np.float64]
    maturities: NDArray[np.float64]
    implied_volatilities: NDArray[np.float64]


@dataclass(frozen=True)
class VolatilitySkew:
    """Linear skew estimate for one maturity slice."""

    maturity: float
    slope: float
    intercept: float
    points: int


@dataclass(frozen=True)
class VolatilitySurfaceAnalyzer:
    """Analyze implied-volatility data across strike and maturity.

    Args:
        data: DataFrame with at least ``strike``, ``maturity``, and
            ``implied_volatility`` columns. If ``moneyness`` is absent and
            ``spot`` is supplied, it is computed as ``strike / spot``.
        spot: Optional underlying spot used to compute moneyness.
    """

    data: pd.DataFrame
    spot: float | None = None

    def __post_init__(self) -> None:
        """Validate and normalize surface data."""
        missing = REQUIRED_SURFACE_COLUMNS.difference(self.data.columns)
        if missing:
            raise ValueError(f"surface data is missing required columns: {sorted(missing)}.")
        if self.spot is not None:
            require_positive(self.spot, "spot")

        normalized = self.data.copy()
        for column in REQUIRED_SURFACE_COLUMNS:
            normalized[column] = normalized[column].astype(float)

        if (normalized["strike"] <= 0.0).any():
            raise ValueError("all strikes must be positive.")
        if (normalized["maturity"] <= 0.0).any():
            raise ValueError("all maturities must be positive.")
        if (normalized["implied_volatility"] < 0.0).any():
            raise ValueError("all implied volatilities must be non-negative.")

        if "moneyness" not in normalized.columns:
            if self.spot is None:
                raise ValueError("spot is required when data does not include moneyness.")
            normalized["moneyness"] = normalized["strike"] / self.spot
        else:
            normalized["moneyness"] = normalized["moneyness"].astype(float)

        normalized = normalized.sort_values(["maturity", "strike"]).reset_index(drop=True)
        object.__setattr__(self, "data", normalized)

    def available_maturities(self) -> NDArray[np.float64]:
        """Return sorted unique maturities."""
        return np.array(sorted(self.data["maturity"].unique()), dtype=np.float64)

    def available_strikes(self) -> NDArray[np.float64]:
        """Return sorted unique strikes."""
        return np.array(sorted(self.data["strike"].unique()), dtype=np.float64)

    def smile(self, maturity: float, tolerance: float = 1e-10) -> pd.DataFrame:
        """Return one volatility smile slice for a given maturity."""
        require_positive(maturity, "maturity")
        require_positive(tolerance, "tolerance")

        mask = np.isclose(self.data["maturity"], maturity, atol=tolerance, rtol=0.0)
        smile = self.data.loc[mask].sort_values("strike").reset_index(drop=True)
        if smile.empty:
            raise ValueError(f"no volatility smile found for maturity={maturity}.")
        return smile

    def term_structure_by_strike(self, strike: float, tolerance: float = 1e-10) -> pd.DataFrame:
        """Return implied volatility across maturities for one strike."""
        require_positive(strike, "strike")
        require_positive(tolerance, "tolerance")

        mask = np.isclose(self.data["strike"], strike, atol=tolerance, rtol=0.0)
        term = self.data.loc[mask].sort_values("maturity").reset_index(drop=True)
        if term.empty:
            raise ValueError(f"no term structure found for strike={strike}.")
        return term

    def atm_term_structure(self) -> pd.DataFrame:
        """Return the closest-to-ATM implied volatility for each maturity."""
        rows: list[pd.Series] = []
        for _, maturity_slice in self.data.groupby("maturity", sort=True):
            atm_index = (maturity_slice["moneyness"] - 1.0).abs().idxmin()
            rows.append(self.data.loc[atm_index])
        return pd.DataFrame(rows).sort_values("maturity").reset_index(drop=True)

    def skew(self, maturity: float) -> VolatilitySkew:
        """Estimate linear volatility skew versus moneyness for one maturity.

        The slope is from ``implied_volatility = intercept + slope * moneyness``.
        Equity-index downside skew often has a negative slope.
        """
        smile = self.smile(maturity)
        if len(smile) < 2:
            raise ValueError("at least two smile points are required to estimate skew.")

        slope, intercept = np.polyfit(
            smile["moneyness"].to_numpy(dtype=np.float64),
            smile["implied_volatility"].to_numpy(dtype=np.float64),
            deg=1,
        )
        return VolatilitySkew(
            maturity=maturity,
            slope=float(slope),
            intercept=float(intercept),
            points=len(smile),
        )

    def surface_grid(self) -> SurfaceGrid:
        """Return a rectangular strike-by-maturity implied-volatility grid."""
        pivot = self.data.pivot_table(
            index="maturity",
            columns="strike",
            values="implied_volatility",
            aggfunc="mean",
        ).sort_index(axis=0).sort_index(axis=1)

        if pivot.isna().any().any():
            raise ValueError(
                "volatility surface grid is incomplete; provide every strike/maturity pair."
            )

        return SurfaceGrid(
            strikes=pivot.columns.to_numpy(dtype=np.float64),
            maturities=pivot.index.to_numpy(dtype=np.float64),
            implied_volatilities=pivot.to_numpy(dtype=np.float64),
        )


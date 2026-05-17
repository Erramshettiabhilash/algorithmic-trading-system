"""Market data containers for pricing models."""

from __future__ import annotations

from dataclasses import dataclass

from utils.validation import require_non_negative, require_positive


@dataclass(frozen=True)
class MarketEnvironment:
    """Market inputs required by vanilla option pricing models.

    Args:
        spot: Current underlying price.
        risk_free_rate: Continuously compounded annual risk-free rate.
        volatility: Annualized volatility as a decimal.
    """

    spot: float
    risk_free_rate: float
    volatility: float

    def __post_init__(self) -> None:
        """Validate market inputs used by pricing and analytics engines."""
        require_positive(self.spot, "spot")
        require_non_negative(self.volatility, "volatility")

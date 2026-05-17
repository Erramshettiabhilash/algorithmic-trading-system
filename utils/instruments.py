"""Financial instrument data structures used across the platform."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from utils.validation import require_positive


class OptionType(StrEnum):
    """Supported vanilla option types."""

    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class EuropeanOption:
    """European vanilla option contract.

    Args:
        strike: Strike price of the option.
        maturity: Time to maturity in years.
        option_type: Call or put.
    """

    strike: float
    maturity: float
    option_type: OptionType

    def __post_init__(self) -> None:
        """Validate option contract inputs."""
        require_positive(self.strike, "strike")
        require_positive(self.maturity, "maturity")

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class OptionType(StrEnum):
    """Supported vanilla option payoff directions."""

    CALL = "call"
    PUT = "put"


@dataclass(frozen=True, slots=True)
class EuropeanOption:
    """A European option exercisable only at maturity."""

    option_type: OptionType
    strike: float
    maturity_years: float

    def __post_init__(self) -> None:
        if self.strike <= 0.0:
            raise ValueError("strike must be positive")
        if self.maturity_years < 0.0:
            raise ValueError("maturity_years must be non-negative")


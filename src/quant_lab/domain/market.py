from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BlackScholesMarket:
    """Market inputs for the Black-Scholes model."""

    spot: float
    rate: float
    volatility: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        if self.spot <= 0.0:
            raise ValueError("spot must be positive")
        if self.volatility < 0.0:
            raise ValueError("volatility must be non-negative")


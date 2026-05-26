from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MonteCarloConfig:
    """Configuration for readable Monte Carlo experiments."""

    n_paths: int
    n_steps: int
    seed: int | None = None
    antithetic: bool = False
    return_paths: bool = True

    def __post_init__(self) -> None:
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative when provided")


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """Summary and samples from a Monte Carlo pricing run."""

    price: float
    standard_error: float
    confidence_interval_95: tuple[float, float]
    terminal_prices: FloatArray
    paths: FloatArray | None = None

    def __post_init__(self) -> None:
        lower_bound, upper_bound = self.confidence_interval_95

        if self.standard_error < 0.0:
            raise ValueError("standard_error must be non-negative")
        if lower_bound > upper_bound:
            raise ValueError("confidence_interval_95 lower bound must not exceed upper bound")
        if self.terminal_prices.ndim != 1:
            raise ValueError("terminal_prices must be a one-dimensional array")
        if self.paths is not None and self.paths.ndim != 2:
            raise ValueError("paths must be a two-dimensional array when provided")

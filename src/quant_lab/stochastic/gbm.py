from __future__ import annotations

import numpy as np

from quant_lab.domain import BlackScholesMarket, FloatArray, MonteCarloConfig
from quant_lab.stochastic.brownian import brownian_increments


def simulate_geometric_brownian_motion(
    market: BlackScholesMarket,
    maturity_years: float,
    config: MonteCarloConfig,
) -> FloatArray:
    """Simulate risk-neutral geometric Brownian motion price paths."""

    if maturity_years < 0.0:
        raise ValueError("maturity_years must be non-negative")

    paths = np.empty((config.n_paths, config.n_steps + 1), dtype=np.float64)
    paths[:, 0] = market.spot

    if maturity_years == 0.0:
        paths[:, 1:] = market.spot
        return paths

    dt = maturity_years / config.n_steps
    increments = brownian_increments(
        n_paths=config.n_paths,
        n_steps=config.n_steps,
        dt=dt,
        seed=config.seed,
        antithetic=config.antithetic,
    )
    drift = (market.rate - market.dividend_yield - 0.5 * market.volatility**2) * dt

    for step_index in range(config.n_steps):
        random_shock = market.volatility * increments[:, step_index]
        growth_factor = np.exp(drift + random_shock)
        paths[:, step_index + 1] = paths[:, step_index] * growth_factor

    return paths

from __future__ import annotations

from math import exp, sqrt

from quant_lab.domain import (
    BlackScholesMarket,
    EuropeanOption,
    FloatArray,
    MonteCarloConfig,
    MonteCarloResult,
)
from quant_lab.pricing.payoffs import european_payoff
from quant_lab.stochastic import simulate_geometric_brownian_motion


def monte_carlo_price(
    option: EuropeanOption,
    market: BlackScholesMarket,
    config: MonteCarloConfig,
) -> MonteCarloResult:
    """Price a European option by discounting simulated risk-neutral payoffs."""

    paths = simulate_geometric_brownian_motion(
        market=market,
        maturity_years=option.maturity_years,
        config=config,
    )
    terminal_prices = paths[:, -1]
    payoffs = european_payoff(option, terminal_prices)
    discounted_payoffs = exp(-market.rate * option.maturity_years) * payoffs

    price = float(discounted_payoffs.mean())
    standard_error = _standard_error(discounted_payoffs)
    confidence_radius = 1.96 * standard_error

    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        confidence_interval_95=(price - confidence_radius, price + confidence_radius),
        terminal_prices=terminal_prices,
        paths=paths if config.return_paths else None,
    )


def _standard_error(values: FloatArray) -> float:
    sample_count = values.size
    if sample_count <= 1:
        return 0.0

    return float(values.std(ddof=1) / sqrt(sample_count))

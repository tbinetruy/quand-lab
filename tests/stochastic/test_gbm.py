from math import exp, isclose

import numpy as np
import pytest

from quant_lab.domain import BlackScholesMarket, MonteCarloConfig
from quant_lab.stochastic import simulate_geometric_brownian_motion


def test_gbm_paths_are_reproducible_with_seed() -> None:
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)
    config = MonteCarloConfig(n_paths=3, n_steps=4, seed=42)

    first = simulate_geometric_brownian_motion(market, maturity_years=1.0, config=config)
    second = simulate_geometric_brownian_motion(market, maturity_years=1.0, config=config)

    np.testing.assert_array_equal(first, second)


def test_gbm_paths_have_expected_shape_and_initial_spot() -> None:
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)
    config = MonteCarloConfig(n_paths=5, n_steps=7, seed=1)

    paths = simulate_geometric_brownian_motion(market, maturity_years=1.0, config=config)

    assert paths.shape == (5, 8)
    np.testing.assert_array_equal(paths[:, 0], np.full(5, market.spot))


def test_gbm_zero_maturity_returns_flat_paths() -> None:
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)
    config = MonteCarloConfig(n_paths=2, n_steps=3, seed=1)

    paths = simulate_geometric_brownian_motion(market, maturity_years=0.0, config=config)

    np.testing.assert_array_equal(paths, np.full((2, 4), market.spot))


def test_gbm_zero_volatility_returns_deterministic_growth() -> None:
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.0, dividend_yield=0.01)
    config = MonteCarloConfig(n_paths=2, n_steps=4, seed=1)

    paths = simulate_geometric_brownian_motion(market, maturity_years=1.0, config=config)
    expected_terminal_spot = market.spot * exp((market.rate - market.dividend_yield) * 1.0)

    assert isclose(paths[0, -1], expected_terminal_spot)
    np.testing.assert_array_equal(paths[0], paths[1])


def test_gbm_terminal_mean_matches_risk_neutral_expectation() -> None:
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2, dividend_yield=0.01)
    config = MonteCarloConfig(n_paths=100_000, n_steps=64, seed=11)

    paths = simulate_geometric_brownian_motion(market, maturity_years=1.0, config=config)
    expected_mean = market.spot * exp((market.rate - market.dividend_yield) * 1.0)

    assert isclose(float(paths[:, -1].mean()), expected_mean, rel_tol=0.01)


def test_gbm_rejects_negative_maturity() -> None:
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)
    config = MonteCarloConfig(n_paths=2, n_steps=3, seed=1)

    with pytest.raises(ValueError, match="maturity_years must be non-negative"):
        simulate_geometric_brownian_motion(market, maturity_years=-1.0, config=config)


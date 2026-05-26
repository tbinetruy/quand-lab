from math import isclose

import pytest

from quant_lab.domain import BlackScholesMarket, EuropeanOption, MonteCarloConfig, OptionType
from quant_lab.pricing import black_scholes_price, monte_carlo_price


def test_monte_carlo_price_is_reproducible_with_seed() -> None:
    option = EuropeanOption(OptionType.CALL, strike=100.0, maturity_years=1.0)
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)
    config = MonteCarloConfig(n_paths=10_000, n_steps=64, seed=42)

    first = monte_carlo_price(option, market, config)
    second = monte_carlo_price(option, market, config)

    assert first.price == second.price
    assert first.standard_error == second.standard_error
    assert first.confidence_interval_95 == second.confidence_interval_95


def test_monte_carlo_price_returns_confidence_interval_and_samples() -> None:
    option = EuropeanOption(OptionType.CALL, strike=100.0, maturity_years=1.0)
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)
    config = MonteCarloConfig(n_paths=1_000, n_steps=16, seed=1)

    result = monte_carlo_price(option, market, config)

    lower_bound, upper_bound = result.confidence_interval_95

    assert lower_bound < result.price < upper_bound
    assert result.standard_error > 0.0
    assert result.terminal_prices.shape == (1_000,)
    assert result.paths is not None
    assert result.paths.shape == (1_000, 17)


def test_monte_carlo_price_can_skip_paths() -> None:
    option = EuropeanOption(OptionType.CALL, strike=100.0, maturity_years=1.0)
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)
    config = MonteCarloConfig(n_paths=1_000, n_steps=16, seed=1, return_paths=False)

    result = monte_carlo_price(option, market, config)

    assert result.paths is None
    assert result.terminal_prices.shape == (1_000,)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_monte_carlo_price_broadly_matches_black_scholes(option_type: OptionType) -> None:
    option = EuropeanOption(option_type, strike=100.0, maturity_years=1.0)
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)
    config = MonteCarloConfig(
        n_paths=150_000,
        n_steps=1,
        seed=7,
        antithetic=True,
        return_paths=False,
    )

    result = monte_carlo_price(option, market, config)
    analytical_price = black_scholes_price(option, market)

    assert isclose(result.price, analytical_price, abs_tol=3.0 * result.standard_error)


def test_monte_carlo_price_handles_single_path_standard_error() -> None:
    option = EuropeanOption(OptionType.CALL, strike=100.0, maturity_years=1.0)
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)
    config = MonteCarloConfig(n_paths=1, n_steps=1, seed=1)

    result = monte_carlo_price(option, market, config)

    assert result.standard_error == 0.0
    assert result.confidence_interval_95 == (result.price, result.price)


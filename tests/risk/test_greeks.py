from math import isclose

import pytest

from quant_lab.domain import BlackScholesMarket, EuropeanOption, OptionType
from quant_lab.risk import black_scholes_greeks, finite_difference_greek
from quant_lab.risk.greeks import GreekName


def test_black_scholes_call_greeks_match_known_values() -> None:
    option = EuropeanOption(OptionType.CALL, strike=100.0, maturity_years=1.0)
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)

    greeks = black_scholes_greeks(option, market)

    assert isclose(greeks.delta, 0.6368, abs_tol=1e-4)
    assert isclose(greeks.gamma, 0.0188, abs_tol=1e-4)
    assert isclose(greeks.vega, 37.5240, abs_tol=1e-3)
    assert isclose(greeks.theta, -6.4140, abs_tol=1e-3)
    assert isclose(greeks.rho, 53.2325, abs_tol=1e-3)


def test_black_scholes_put_greeks_match_known_values() -> None:
    option = EuropeanOption(OptionType.PUT, strike=100.0, maturity_years=1.0)
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)

    greeks = black_scholes_greeks(option, market)

    assert isclose(greeks.delta, -0.3632, abs_tol=1e-4)
    assert isclose(greeks.gamma, 0.0188, abs_tol=1e-4)
    assert isclose(greeks.vega, 37.5240, abs_tol=1e-3)
    assert isclose(greeks.theta, -1.6579, abs_tol=1e-3)
    assert isclose(greeks.rho, -41.8905, abs_tol=1e-3)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("greek", ["delta", "gamma", "vega", "theta", "rho"])
def test_analytical_greeks_match_finite_differences(
    option_type: OptionType,
    greek: GreekName,
) -> None:
    option = EuropeanOption(option_type, strike=105.0, maturity_years=1.25)
    market = BlackScholesMarket(
        spot=100.0,
        rate=0.04,
        volatility=0.25,
        dividend_yield=0.01,
    )

    analytical = getattr(black_scholes_greeks(option, market), greek)
    finite_difference = finite_difference_greek(option, market, greek)

    assert isclose(analytical, finite_difference, rel_tol=2e-3, abs_tol=2e-3)


def test_analytical_greeks_reject_zero_maturity() -> None:
    option = EuropeanOption(OptionType.CALL, strike=100.0, maturity_years=0.0)
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)

    with pytest.raises(ValueError, match="positive maturity"):
        black_scholes_greeks(option, market)


def test_analytical_greeks_reject_zero_volatility() -> None:
    option = EuropeanOption(OptionType.CALL, strike=100.0, maturity_years=1.0)
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.0)

    with pytest.raises(ValueError, match="positive volatility"):
        black_scholes_greeks(option, market)

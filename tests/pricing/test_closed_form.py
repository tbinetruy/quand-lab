from math import exp, isclose

import pytest

from quant_lab.domain import BlackScholesMarket, EuropeanOption, OptionType
from quant_lab.pricing import black_scholes_price


def test_black_scholes_call_matches_known_value() -> None:
    option = EuropeanOption(
        option_type=OptionType.CALL,
        strike=100.0,
        maturity_years=1.0,
    )
    market = BlackScholesMarket(
        spot=100.0,
        rate=0.05,
        volatility=0.2,
    )

    assert isclose(black_scholes_price(option, market), 10.4506, abs_tol=1e-4)


def test_black_scholes_put_matches_known_value() -> None:
    option = EuropeanOption(
        option_type=OptionType.PUT,
        strike=100.0,
        maturity_years=1.0,
    )
    market = BlackScholesMarket(
        spot=100.0,
        rate=0.05,
        volatility=0.2,
    )

    assert isclose(black_scholes_price(option, market), 5.5735, abs_tol=1e-4)


def test_black_scholes_supports_dividend_yield() -> None:
    option = EuropeanOption(
        option_type=OptionType.CALL,
        strike=100.0,
        maturity_years=1.0,
    )
    market = BlackScholesMarket(
        spot=100.0,
        rate=0.05,
        volatility=0.2,
        dividend_yield=0.02,
    )

    assert isclose(black_scholes_price(option, market), 9.2270, abs_tol=1e-4)


def test_black_scholes_put_call_parity_holds() -> None:
    call = EuropeanOption(
        option_type=OptionType.CALL,
        strike=95.0,
        maturity_years=1.5,
    )
    put = EuropeanOption(
        option_type=OptionType.PUT,
        strike=95.0,
        maturity_years=1.5,
    )
    market = BlackScholesMarket(
        spot=105.0,
        rate=0.04,
        volatility=0.25,
        dividend_yield=0.01,
    )

    call_price = black_scholes_price(call, market)
    put_price = black_scholes_price(put, market)
    discounted_spot = market.spot * exp(-market.dividend_yield * call.maturity_years)
    discounted_strike = call.strike * exp(-market.rate * call.maturity_years)

    assert isclose(call_price - put_price, discounted_spot - discounted_strike, abs_tol=1e-12)


@pytest.mark.parametrize(
    ("option_type", "expected_price"),
    [
        (OptionType.CALL, 20.0),
        (OptionType.PUT, 0.0),
    ],
)
def test_black_scholes_at_expiry_returns_intrinsic_value(
    option_type: OptionType,
    expected_price: float,
) -> None:
    option = EuropeanOption(
        option_type=option_type,
        strike=80.0,
        maturity_years=0.0,
    )
    market = BlackScholesMarket(
        spot=100.0,
        rate=0.05,
        volatility=0.2,
    )

    assert black_scholes_price(option, market) == expected_price


def test_black_scholes_zero_volatility_returns_discounted_deterministic_payoff() -> None:
    option = EuropeanOption(
        option_type=OptionType.CALL,
        strike=100.0,
        maturity_years=1.0,
    )
    market = BlackScholesMarket(
        spot=100.0,
        rate=0.05,
        volatility=0.0,
    )

    assert isclose(black_scholes_price(option, market), 4.8771, abs_tol=1e-4)

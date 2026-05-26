from __future__ import annotations

from math import isclose

import pytest

from quant_lab.domain import BlackScholesMarket, EuropeanOption, OptionType
from quant_lab.pricing import black_scholes_price
from quant_lab.risk import (
    CashPosition,
    OptionPosition,
    StockPosition,
    estimate_portfolio_value_change,
    option_portfolio_payoff,
    value_option_portfolio,
)
from quant_lab.risk.greeks import black_scholes_greeks


def test_portfolio_value_is_sum_of_signed_option_values() -> None:
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)
    call = EuropeanOption(OptionType.CALL, strike=100.0, maturity_years=1.0)
    put = EuropeanOption(OptionType.PUT, strike=100.0, maturity_years=1.0)
    positions = [
        OptionPosition(call, quantity=1.0),
        OptionPosition(put, quantity=1.0),
    ]

    valuation = value_option_portfolio(positions, market)

    expected_value = black_scholes_price(call, market) + black_scholes_price(put, market)
    assert isclose(valuation.total_value, expected_value, abs_tol=1e-12)


def test_portfolio_greeks_are_quantity_weighted() -> None:
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)
    call = EuropeanOption(OptionType.CALL, strike=100.0, maturity_years=1.0)
    put = EuropeanOption(OptionType.PUT, strike=95.0, maturity_years=1.0)
    positions = [
        OptionPosition(call, quantity=2.0),
        OptionPosition(put, quantity=-1.0),
    ]

    valuation = value_option_portfolio(positions, market)
    call_greeks = black_scholes_greeks(call, market)
    put_greeks = black_scholes_greeks(put, market)

    assert isclose(
        valuation.total_greeks.delta,
        2.0 * call_greeks.delta - put_greeks.delta,
        abs_tol=1e-12,
    )
    assert isclose(
        valuation.total_greeks.gamma,
        2.0 * call_greeks.gamma - put_greeks.gamma,
        abs_tol=1e-12,
    )
    assert isclose(
        valuation.total_greeks.vega,
        2.0 * call_greeks.vega - put_greeks.vega,
        abs_tol=1e-12,
    )


def test_portfolio_payoff_respects_long_and_short_positions() -> None:
    call = EuropeanOption(OptionType.CALL, strike=100.0, maturity_years=1.0)
    put = EuropeanOption(OptionType.PUT, strike=100.0, maturity_years=1.0)
    positions = [
        OptionPosition(call, quantity=1.0),
        OptionPosition(put, quantity=-1.0),
    ]

    assert isclose(option_portfolio_payoff(positions, terminal_spot=120.0), 20.0)
    assert isclose(option_portfolio_payoff(positions, terminal_spot=80.0), -20.0)


def test_stock_and_cash_positions_are_included_in_value_and_delta() -> None:
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)
    call = EuropeanOption(OptionType.CALL, strike=100.0, maturity_years=1.0)
    positions = [
        StockPosition(quantity=1.0),
        OptionPosition(call, quantity=-1.0),
        CashPosition(amount=5.0),
    ]

    valuation = value_option_portfolio(positions, market)
    call_price = black_scholes_price(call, market)
    call_greeks = black_scholes_greeks(call, market)

    assert isclose(valuation.total_value, 100.0 - call_price + 5.0, abs_tol=1e-12)
    assert isclose(valuation.total_greeks.delta, 1.0 - call_greeks.delta, abs_tol=1e-12)
    assert isclose(option_portfolio_payoff(positions, terminal_spot=120.0), 105.0)


def test_greek_value_change_matches_small_spot_reprice() -> None:
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)
    call = EuropeanOption(OptionType.CALL, strike=100.0, maturity_years=1.0)
    positions = [OptionPosition(call, quantity=1.0)]
    valuation = value_option_portfolio(positions, market)
    shocked_market = BlackScholesMarket(spot=100.1, rate=0.05, volatility=0.2)
    shocked_valuation = value_option_portfolio(positions, shocked_market)

    estimated_change = estimate_portfolio_value_change(valuation, spot_change=0.1)

    assert isclose(
        estimated_change,
        shocked_valuation.total_value - valuation.total_value,
        rel_tol=1e-3,
    )


def test_empty_portfolio_is_rejected() -> None:
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)

    with pytest.raises(ValueError, match="portfolio must contain at least one position"):
        value_option_portfolio([], market)

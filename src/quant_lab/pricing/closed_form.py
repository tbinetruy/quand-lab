from __future__ import annotations

from math import erf, exp, log, sqrt

from quant_lab.domain import BlackScholesMarket, EuropeanOption, OptionType


def black_scholes_price(option: EuropeanOption, market: BlackScholesMarket) -> float:
    """Price a European option with the Black-Scholes closed-form formula."""

    if option.maturity_years == 0.0:
        return _intrinsic_value(option, market.spot)

    if market.volatility == 0.0:
        return _deterministic_discounted_payoff(option, market)

    d1 = _d1(option, market)
    d2 = d1 - market.volatility * sqrt(option.maturity_years)

    discounted_spot = market.spot * exp(-market.dividend_yield * option.maturity_years)
    discounted_strike = option.strike * exp(-market.rate * option.maturity_years)

    if option.option_type is OptionType.CALL:
        expected_spot = discounted_spot * _standard_normal_cdf(d1)
        expected_strike = discounted_strike * _standard_normal_cdf(d2)
        return expected_spot - expected_strike

    expected_strike = discounted_strike * _standard_normal_cdf(-d2)
    expected_spot = discounted_spot * _standard_normal_cdf(-d1)
    return expected_strike - expected_spot


def _d1(option: EuropeanOption, market: BlackScholesMarket) -> float:
    time_to_maturity = option.maturity_years
    volatility_time = market.volatility * sqrt(time_to_maturity)
    forward_log_moneyness = log(market.spot / option.strike)
    carry_adjusted_drift = (
        market.rate - market.dividend_yield + 0.5 * market.volatility**2
    ) * time_to_maturity

    return (forward_log_moneyness + carry_adjusted_drift) / volatility_time


def _deterministic_discounted_payoff(
    option: EuropeanOption,
    market: BlackScholesMarket,
) -> float:
    terminal_spot = market.spot * exp(
        (market.rate - market.dividend_yield) * option.maturity_years
    )
    discounted_payoff = exp(-market.rate * option.maturity_years) * _intrinsic_value(
        option,
        terminal_spot,
    )

    return discounted_payoff


def _intrinsic_value(option: EuropeanOption, spot: float) -> float:
    if option.option_type is OptionType.CALL:
        return max(spot - option.strike, 0.0)

    return max(option.strike - spot, 0.0)


def _standard_normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))

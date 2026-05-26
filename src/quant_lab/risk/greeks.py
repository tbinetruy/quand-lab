from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from typing import Literal

from quant_lab.domain import BlackScholesMarket, EuropeanOption, OptionType
from quant_lab.pricing import black_scholes_price

GreekName = Literal["delta", "gamma", "vega", "theta", "rho"]


@dataclass(frozen=True, slots=True)
class BlackScholesGreeks:
    """Black-Scholes sensitivities for a European option."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def black_scholes_greeks(
    option: EuropeanOption,
    market: BlackScholesMarket,
) -> BlackScholesGreeks:
    """Calculate analytical Black-Scholes Greeks."""

    _ensure_smooth_greeks(option, market)

    time_to_maturity = option.maturity_years
    volatility_time = market.volatility * sqrt(time_to_maturity)
    d1 = _d1(option, market)
    d2 = d1 - volatility_time
    normal_density_d1 = _standard_normal_pdf(d1)
    discounted_spot_factor = exp(-market.dividend_yield * time_to_maturity)
    discounted_strike_factor = exp(-market.rate * time_to_maturity)

    gamma = (
        discounted_spot_factor
        * normal_density_d1
        / (market.spot * market.volatility * sqrt(time_to_maturity))
    )
    vega = market.spot * discounted_spot_factor * normal_density_d1 * sqrt(time_to_maturity)

    if option.option_type is OptionType.CALL:
        delta = discounted_spot_factor * _standard_normal_cdf(d1)
        theta = (
            -market.spot
            * discounted_spot_factor
            * normal_density_d1
            * market.volatility
            / (2.0 * sqrt(time_to_maturity))
            - market.rate * option.strike * discounted_strike_factor * _standard_normal_cdf(d2)
            + market.dividend_yield
            * market.spot
            * discounted_spot_factor
            * _standard_normal_cdf(d1)
        )
        rho = (
            option.strike
            * time_to_maturity
            * discounted_strike_factor
            * _standard_normal_cdf(d2)
        )
    else:
        delta = -discounted_spot_factor * _standard_normal_cdf(-d1)
        theta = (
            -market.spot
            * discounted_spot_factor
            * normal_density_d1
            * market.volatility
            / (2.0 * sqrt(time_to_maturity))
            + market.rate * option.strike * discounted_strike_factor * _standard_normal_cdf(-d2)
            - market.dividend_yield
            * market.spot
            * discounted_spot_factor
            * _standard_normal_cdf(-d1)
        )
        rho = (
            -option.strike
            * time_to_maturity
            * discounted_strike_factor
            * _standard_normal_cdf(-d2)
        )

    return BlackScholesGreeks(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
    )


def finite_difference_greek(
    option: EuropeanOption,
    market: BlackScholesMarket,
    greek: GreekName,
    *,
    bump: float | None = None,
) -> float:
    """Approximate a Greek by central finite differences."""

    if greek == "delta":
        spot_bump = bump or max(market.spot * 1e-4, 1e-4)
        return _central_difference_spot(option, market, spot_bump)
    if greek == "gamma":
        spot_bump = bump or max(market.spot * 1e-3, 1e-3)
        return _central_second_difference_spot(option, market, spot_bump)
    if greek == "vega":
        volatility_bump = bump or 1e-4
        return _central_difference_volatility(option, market, volatility_bump)
    if greek == "theta":
        time_bump = bump or min(max(option.maturity_years * 1e-4, 1e-5), 1e-3)
        return _central_difference_maturity(option, market, time_bump)
    if greek == "rho":
        rate_bump = bump or 1e-4
        return _central_difference_rate(option, market, rate_bump)

    raise ValueError(f"unsupported greek: {greek}")


def _central_difference_spot(
    option: EuropeanOption,
    market: BlackScholesMarket,
    bump: float,
) -> float:
    lower_spot = max(market.spot - bump, 1e-12)
    upper_spot = market.spot + bump

    lower = black_scholes_price(option, _replace_market(market, spot=lower_spot))
    upper = black_scholes_price(option, _replace_market(market, spot=upper_spot))
    return (upper - lower) / (upper_spot - lower_spot)


def _central_second_difference_spot(
    option: EuropeanOption,
    market: BlackScholesMarket,
    bump: float,
) -> float:
    lower_spot = max(market.spot - bump, 1e-12)
    upper_spot = market.spot + bump
    denominator_bump = upper_spot - market.spot

    lower = black_scholes_price(option, _replace_market(market, spot=lower_spot))
    center = black_scholes_price(option, market)
    upper = black_scholes_price(option, _replace_market(market, spot=upper_spot))
    return (upper - 2.0 * center + lower) / denominator_bump**2


def _central_difference_volatility(
    option: EuropeanOption,
    market: BlackScholesMarket,
    bump: float,
) -> float:
    lower_volatility = max(market.volatility - bump, 1e-12)
    upper_volatility = market.volatility + bump

    lower = black_scholes_price(option, _replace_market(market, volatility=lower_volatility))
    upper = black_scholes_price(option, _replace_market(market, volatility=upper_volatility))
    return (upper - lower) / (upper_volatility - lower_volatility)


def _central_difference_maturity(
    option: EuropeanOption,
    market: BlackScholesMarket,
    bump: float,
) -> float:
    lower_maturity = max(option.maturity_years - bump, 1e-12)
    upper_maturity = option.maturity_years + bump

    lower = black_scholes_price(_replace_option(option, maturity_years=lower_maturity), market)
    upper = black_scholes_price(_replace_option(option, maturity_years=upper_maturity), market)

    # Black-Scholes theta is conventionally dV/dt, where t moves forward and T decreases.
    return -(upper - lower) / (upper_maturity - lower_maturity)


def _central_difference_rate(
    option: EuropeanOption,
    market: BlackScholesMarket,
    bump: float,
) -> float:
    lower = black_scholes_price(option, _replace_market(market, rate=market.rate - bump))
    upper = black_scholes_price(option, _replace_market(market, rate=market.rate + bump))
    return (upper - lower) / (2.0 * bump)


def _ensure_smooth_greeks(option: EuropeanOption, market: BlackScholesMarket) -> None:
    if option.maturity_years <= 0.0:
        raise ValueError("analytical Greeks require positive maturity")
    if market.volatility <= 0.0:
        raise ValueError("analytical Greeks require positive volatility")


def _replace_market(
    market: BlackScholesMarket,
    *,
    spot: float | None = None,
    rate: float | None = None,
    volatility: float | None = None,
) -> BlackScholesMarket:
    return BlackScholesMarket(
        spot=market.spot if spot is None else spot,
        rate=market.rate if rate is None else rate,
        volatility=market.volatility if volatility is None else volatility,
        dividend_yield=market.dividend_yield,
    )


def _replace_option(
    option: EuropeanOption,
    *,
    maturity_years: float,
) -> EuropeanOption:
    return EuropeanOption(
        option_type=option.option_type,
        strike=option.strike,
        maturity_years=maturity_years,
    )


def _d1(option: EuropeanOption, market: BlackScholesMarket) -> float:
    time_to_maturity = option.maturity_years
    volatility_time = market.volatility * sqrt(time_to_maturity)
    forward_log_moneyness = log(market.spot / option.strike)
    carry_adjusted_drift = (
        market.rate - market.dividend_yield + 0.5 * market.volatility**2
    ) * time_to_maturity

    return (forward_log_moneyness + carry_adjusted_drift) / volatility_time


def _standard_normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _standard_normal_pdf(value: float) -> float:
    return exp(-0.5 * value**2) / sqrt(2.0 * pi)


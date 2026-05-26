from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import exp

from quant_lab.domain import BlackScholesMarket, EuropeanOption, OptionType
from quant_lab.pricing import black_scholes_price
from quant_lab.risk.greeks import BlackScholesGreeks, black_scholes_greeks


@dataclass(frozen=True, slots=True)
class OptionPosition:
    """A signed position in one European option contract."""

    option: EuropeanOption
    quantity: float
    contract_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.quantity == 0.0:
            raise ValueError("quantity must be non-zero")
        if self.contract_multiplier <= 0.0:
            raise ValueError("contract_multiplier must be positive")


@dataclass(frozen=True, slots=True)
class StockPosition:
    """A signed position in the underlying asset."""

    quantity: float

    def __post_init__(self) -> None:
        if self.quantity == 0.0:
            raise ValueError("quantity must be non-zero")


@dataclass(frozen=True, slots=True)
class CashPosition:
    """A cash balance held at today's value."""

    amount: float

    def __post_init__(self) -> None:
        if self.amount == 0.0:
            raise ValueError("amount must be non-zero")


PortfolioPosition = OptionPosition | StockPosition | CashPosition


@dataclass(frozen=True, slots=True)
class PositionValuation:
    """Current value and risk contribution for one portfolio position."""

    position: PortfolioPosition
    label: str
    unit_price: float
    market_value: float
    unit_greeks: BlackScholesGreeks
    total_greeks: BlackScholesGreeks


@dataclass(frozen=True, slots=True)
class PortfolioValuation:
    """Aggregated value and Greeks for a list of option positions."""

    positions: tuple[PositionValuation, ...]
    total_value: float
    total_greeks: BlackScholesGreeks


def value_option_position(
    position: OptionPosition,
    market: BlackScholesMarket,
) -> PositionValuation:
    """Price one option position and scale its Greeks by signed quantity."""

    unit_price = black_scholes_price(position.option, market)
    scaling = position.quantity * position.contract_multiplier
    unit_greeks = black_scholes_greeks(position.option, market)

    return PositionValuation(
        position=position,
        label=_option_label(position.option),
        unit_price=unit_price,
        market_value=scaling * unit_price,
        unit_greeks=unit_greeks,
        total_greeks=_scale_greeks(unit_greeks, scaling),
    )


def value_stock_position(
    position: StockPosition,
    market: BlackScholesMarket,
) -> PositionValuation:
    """Value an underlying position and represent its spot exposure as delta."""

    unit_greeks = BlackScholesGreeks(delta=1.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    return PositionValuation(
        position=position,
        label="Stock",
        unit_price=market.spot,
        market_value=position.quantity * market.spot,
        unit_greeks=unit_greeks,
        total_greeks=_scale_greeks(unit_greeks, position.quantity),
    )


def value_cash_position(
    position: CashPosition,
) -> PositionValuation:
    """Value a cash balance held today."""

    zero_greeks = BlackScholesGreeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    return PositionValuation(
        position=position,
        label="Cash",
        unit_price=1.0,
        market_value=position.amount,
        unit_greeks=zero_greeks,
        total_greeks=zero_greeks,
    )


def value_position(
    position: PortfolioPosition,
    market: BlackScholesMarket,
) -> PositionValuation:
    """Value one supported portfolio position."""

    if isinstance(position, OptionPosition):
        return value_option_position(position, market)
    if isinstance(position, StockPosition):
        return value_stock_position(position, market)

    return value_cash_position(position)


def value_option_portfolio(
    positions: Sequence[PortfolioPosition],
    market: BlackScholesMarket,
) -> PortfolioValuation:
    """Aggregate Black-Scholes values and Greeks for a portfolio."""

    if len(positions) == 0:
        raise ValueError("portfolio must contain at least one position")

    valuations = tuple(value_position(position, market) for position in positions)

    return PortfolioValuation(
        positions=valuations,
        total_value=sum(valuation.market_value for valuation in valuations),
        total_greeks=_sum_greeks(valuation.total_greeks for valuation in valuations),
    )


def option_position_payoff(position: OptionPosition, terminal_spot: float) -> float:
    """Evaluate the signed payoff of one option position at maturity."""

    if terminal_spot < 0.0:
        raise ValueError("terminal_spot must be non-negative")

    scaling = position.quantity * position.contract_multiplier
    payoff = _option_intrinsic_value(position.option, terminal_spot)
    return scaling * payoff


def option_portfolio_payoff(
    positions: Sequence[PortfolioPosition],
    terminal_spot: float,
) -> float:
    """Evaluate the total signed payoff of a portfolio at option maturity."""

    if len(positions) == 0:
        raise ValueError("portfolio must contain at least one position")

    return sum(_position_payoff(position, terminal_spot) for position in positions)


def reprice_option_portfolio(
    positions: Sequence[PortfolioPosition],
    market: BlackScholesMarket,
    *,
    spot: float | None = None,
    volatility: float | None = None,
    maturity_years: float | None = None,
) -> PortfolioValuation:
    """Reprice a portfolio after replacing selected market or maturity inputs."""

    shocked_market = BlackScholesMarket(
        spot=market.spot if spot is None else spot,
        rate=market.rate,
        volatility=market.volatility if volatility is None else volatility,
        dividend_yield=market.dividend_yield,
    )
    shocked_positions = tuple(
        _replace_position_maturity(position, maturity_years) for position in positions
    )

    return value_option_portfolio(shocked_positions, shocked_market)


def estimate_portfolio_value_change(
    valuation: PortfolioValuation,
    *,
    spot_change: float = 0.0,
    volatility_change: float = 0.0,
    years_elapsed: float = 0.0,
    rate_change: float = 0.0,
) -> float:
    """Estimate value change from aggregated Greeks around the current state."""

    greeks = valuation.total_greeks
    return (
        greeks.delta * spot_change
        + 0.5 * greeks.gamma * spot_change**2
        + greeks.vega * volatility_change
        + greeks.theta * years_elapsed
        + greeks.rho * rate_change
    )


def present_value_of_cash(
    cash_at_maturity: float,
    market: BlackScholesMarket,
    maturity: float,
) -> float:
    """Discount a deterministic cashflow paid at strategy maturity."""

    if maturity < 0.0:
        raise ValueError("maturity must be non-negative")

    return cash_at_maturity * exp(-market.rate * maturity)


def _replace_position_maturity(
    position: PortfolioPosition,
    maturity_years: float | None,
) -> PortfolioPosition:
    if maturity_years is None or not isinstance(position, OptionPosition):
        return position

    return OptionPosition(
        option=EuropeanOption(
            option_type=position.option.option_type,
            strike=position.option.strike,
            maturity_years=maturity_years,
        ),
        quantity=position.quantity,
        contract_multiplier=position.contract_multiplier,
    )


def _option_intrinsic_value(option: EuropeanOption, spot: float) -> float:
    if option.option_type is OptionType.CALL:
        return max(spot - option.strike, 0.0)

    return max(option.strike - spot, 0.0)


def _position_payoff(position: PortfolioPosition, terminal_spot: float) -> float:
    if isinstance(position, OptionPosition):
        return option_position_payoff(position, terminal_spot)
    if isinstance(position, StockPosition):
        return position.quantity * terminal_spot

    return position.amount


def _option_label(option: EuropeanOption) -> str:
    option_name = option.option_type.value.title()
    return f"{option_name} K={option.strike:g}"


def _scale_greeks(greeks: BlackScholesGreeks, scale: float) -> BlackScholesGreeks:
    return BlackScholesGreeks(
        delta=scale * greeks.delta,
        gamma=scale * greeks.gamma,
        vega=scale * greeks.vega,
        theta=scale * greeks.theta,
        rho=scale * greeks.rho,
    )


def _sum_greeks(greeks: Iterable[BlackScholesGreeks]) -> BlackScholesGreeks:
    total_delta = 0.0
    total_gamma = 0.0
    total_vega = 0.0
    total_theta = 0.0
    total_rho = 0.0

    for greek in greeks:
        total_delta += greek.delta
        total_gamma += greek.gamma
        total_vega += greek.vega
        total_theta += greek.theta
        total_rho += greek.rho

    return BlackScholesGreeks(
        delta=total_delta,
        gamma=total_gamma,
        vega=total_vega,
        theta=total_theta,
        rho=total_rho,
    )

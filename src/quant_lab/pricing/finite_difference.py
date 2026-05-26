from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import exp

import numpy as np

from quant_lab.domain import BlackScholesMarket, EuropeanOption, FloatArray, OptionType
from quant_lab.pricing.payoffs import european_payoff


class FiniteDifferenceMethod(StrEnum):
    """Supported finite-difference schemes for the Black-Scholes PDE."""

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    CRANK_NICOLSON = "crank_nicolson"


@dataclass(frozen=True, slots=True)
class FiniteDifferenceConfig:
    """Grid configuration for finite-difference option pricing."""

    n_spot_steps: int = 160
    n_time_steps: int = 400
    spot_max_multiplier: float = 3.0
    method: FiniteDifferenceMethod = FiniteDifferenceMethod.CRANK_NICOLSON

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", FiniteDifferenceMethod(self.method))

        if self.n_spot_steps < 3:
            raise ValueError("n_spot_steps must be at least 3")
        if self.n_time_steps < 1:
            raise ValueError("n_time_steps must be positive")
        if self.spot_max_multiplier <= 1.0:
            raise ValueError("spot_max_multiplier must be greater than 1")


@dataclass(frozen=True, slots=True)
class FiniteDifferenceResult:
    """Full finite-difference grid and interpolated price."""

    price: float
    spot_grid: FloatArray
    time_to_expiry_grid: FloatArray
    values: FloatArray
    method: FiniteDifferenceMethod
    stability_ratio: float


def finite_difference_price(
    option: EuropeanOption,
    market: BlackScholesMarket,
    config: FiniteDifferenceConfig | None = None,
) -> FiniteDifferenceResult:
    """Price a European option by solving the Black-Scholes PDE on a grid."""

    solver_config = config or FiniteDifferenceConfig()
    if option.maturity_years < 0.0:
        raise ValueError("maturity_years must be non-negative")

    spot_max = max(market.spot, option.strike) * solver_config.spot_max_multiplier
    spot_grid = np.linspace(0.0, spot_max, solver_config.n_spot_steps + 1, dtype=np.float64)
    time_grid = np.linspace(
        0.0,
        option.maturity_years,
        solver_config.n_time_steps + 1,
        dtype=np.float64,
    )
    values = np.empty((solver_config.n_time_steps + 1, solver_config.n_spot_steps + 1))
    values[0, :] = european_payoff(option, spot_grid)

    if option.maturity_years == 0.0:
        values[:, :] = values[0, :]
        return FiniteDifferenceResult(
            price=_interpolate_on_grid(market.spot, spot_grid, values[0]),
            spot_grid=spot_grid,
            time_to_expiry_grid=time_grid,
            values=values,
            method=solver_config.method,
            stability_ratio=0.0,
        )

    dt = option.maturity_years / solver_config.n_time_steps
    stability_ratio = market.volatility**2 * solver_config.n_spot_steps**2 * dt

    for time_index in range(solver_config.n_time_steps):
        tau_next = time_grid[time_index + 1]
        values[time_index + 1, 0] = _lower_boundary(option, market, tau_next)
        values[time_index + 1, -1] = _upper_boundary(option, market, spot_max, tau_next)

        if solver_config.method is FiniteDifferenceMethod.EXPLICIT:
            _step_explicit(values, time_index, market, solver_config.n_spot_steps, dt)
        elif solver_config.method is FiniteDifferenceMethod.IMPLICIT:
            _step_implicit(values, time_index, market, dt)
        else:
            _step_crank_nicolson(values, time_index, market, dt)

    price = _interpolate_on_grid(market.spot, spot_grid, values[-1])
    return FiniteDifferenceResult(
        price=price,
        spot_grid=spot_grid,
        time_to_expiry_grid=time_grid,
        values=values,
        method=solver_config.method,
        stability_ratio=stability_ratio,
    )


def _step_explicit(
    values: FloatArray,
    time_index: int,
    market: BlackScholesMarket,
    n_spot_steps: int,
    dt: float,
) -> None:
    for spot_index in range(1, n_spot_steps):
        lower, center, upper = _explicit_coefficients(spot_index, market, dt)
        values[time_index + 1, spot_index] = (
            lower * values[time_index, spot_index - 1]
            + center * values[time_index, spot_index]
            + upper * values[time_index, spot_index + 1]
        )


def _step_implicit(
    values: FloatArray,
    time_index: int,
    market: BlackScholesMarket,
    dt: float,
) -> None:
    system = _implicit_system(
        market=market,
        n_spot_steps=len(values[time_index]) - 1,
        dt=dt,
        theta=1.0,
    )
    rhs = values[time_index, 1:-1].copy()
    rhs[0] -= system.lower[0] * values[time_index + 1, 0]
    rhs[-1] -= system.upper[-1] * values[time_index + 1, -1]
    values[time_index + 1, 1:-1] = _solve_tridiagonal(
        system.lower,
        system.diagonal,
        system.upper,
        rhs,
    )


def _step_crank_nicolson(
    values: FloatArray,
    time_index: int,
    market: BlackScholesMarket,
    dt: float,
) -> None:
    system = _implicit_system(
        market=market,
        n_spot_steps=len(values[time_index]) - 1,
        dt=dt,
        theta=0.5,
    )
    rhs = values[time_index, 1:-1].copy()

    for interior_index, spot_index in enumerate(range(1, len(values[time_index]) - 1)):
        lower, center, upper = _operator_coefficients(spot_index, market)
        rhs[interior_index] += 0.5 * dt * (
            lower * values[time_index, spot_index - 1]
            + center * values[time_index, spot_index]
            + upper * values[time_index, spot_index + 1]
        )

    rhs[0] -= system.lower[0] * values[time_index + 1, 0]
    rhs[-1] -= system.upper[-1] * values[time_index + 1, -1]
    values[time_index + 1, 1:-1] = _solve_tridiagonal(
        system.lower,
        system.diagonal,
        system.upper,
        rhs,
    )


@dataclass(frozen=True, slots=True)
class _TridiagonalSystem:
    lower: FloatArray
    diagonal: FloatArray
    upper: FloatArray


def _implicit_system(
    *,
    market: BlackScholesMarket,
    n_spot_steps: int,
    dt: float,
    theta: float,
) -> _TridiagonalSystem:
    n_interior = n_spot_steps - 1
    lower = np.empty(n_interior, dtype=np.float64)
    diagonal = np.empty(n_interior, dtype=np.float64)
    upper = np.empty(n_interior, dtype=np.float64)

    for interior_index, spot_index in enumerate(range(1, n_spot_steps)):
        lower_coeff, center_coeff, upper_coeff = _operator_coefficients(spot_index, market)
        lower[interior_index] = -theta * dt * lower_coeff
        diagonal[interior_index] = 1.0 - theta * dt * center_coeff
        upper[interior_index] = -theta * dt * upper_coeff

    return _TridiagonalSystem(lower=lower, diagonal=diagonal, upper=upper)


def _operator_coefficients(
    spot_index: int,
    market: BlackScholesMarket,
) -> tuple[float, float, float]:
    carry = market.rate - market.dividend_yield
    variance_term = market.volatility**2 * spot_index**2
    lower = 0.5 * (variance_term - carry * spot_index)
    center = -(variance_term + market.rate)
    upper = 0.5 * (variance_term + carry * spot_index)
    return lower, center, upper


def _explicit_coefficients(
    spot_index: int,
    market: BlackScholesMarket,
    dt: float,
) -> tuple[float, float, float]:
    lower, center, upper = _operator_coefficients(spot_index, market)
    return dt * lower, 1.0 + dt * center, dt * upper


def _solve_tridiagonal(
    lower: FloatArray,
    diagonal: FloatArray,
    upper: FloatArray,
    rhs: FloatArray,
) -> FloatArray:
    modified_diagonal = diagonal.copy()
    modified_rhs = rhs.copy()

    for row in range(1, len(diagonal)):
        multiplier = lower[row] / modified_diagonal[row - 1]
        modified_diagonal[row] -= multiplier * upper[row - 1]
        modified_rhs[row] -= multiplier * modified_rhs[row - 1]

    solution = np.empty_like(rhs)
    solution[-1] = modified_rhs[-1] / modified_diagonal[-1]

    for row in range(len(diagonal) - 2, -1, -1):
        solution[row] = (
            modified_rhs[row] - upper[row] * solution[row + 1]
        ) / modified_diagonal[row]

    return solution


def _lower_boundary(option: EuropeanOption, market: BlackScholesMarket, tau: float) -> float:
    if option.option_type is OptionType.CALL:
        return 0.0
    return option.strike * exp(-market.rate * tau)


def _upper_boundary(
    option: EuropeanOption,
    market: BlackScholesMarket,
    spot_max: float,
    tau: float,
) -> float:
    if option.option_type is OptionType.CALL:
        return spot_max * exp(-market.dividend_yield * tau) - option.strike * exp(
            -market.rate * tau
        )
    return 0.0


def _interpolate_on_grid(spot: float, spot_grid: FloatArray, values: FloatArray) -> float:
    if spot <= spot_grid[0]:
        return float(values[0])
    if spot >= spot_grid[-1]:
        return float(values[-1])

    for index in range(1, len(spot_grid)):
        upper_spot = float(spot_grid[index])
        if spot <= upper_spot:
            lower_spot = float(spot_grid[index - 1])
            lower_value = float(values[index - 1])
            upper_value = float(values[index])
            weight = (spot - lower_spot) / (upper_spot - lower_spot)
            return lower_value + weight * (upper_value - lower_value)

    return float(values[-1])

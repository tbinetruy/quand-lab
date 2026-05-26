from math import isclose

import numpy as np
import pytest

from quant_lab.domain import BlackScholesMarket, EuropeanOption, OptionType
from quant_lab.pricing import (
    FiniteDifferenceConfig,
    FiniteDifferenceMethod,
    black_scholes_price,
    finite_difference_price,
)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize(
    ("method", "n_time_steps", "tolerance"),
    [
        (FiniteDifferenceMethod.EXPLICIT, 1_200, 0.08),
        (FiniteDifferenceMethod.IMPLICIT, 160, 0.08),
        (FiniteDifferenceMethod.CRANK_NICOLSON, 160, 0.05),
    ],
)
def test_finite_difference_price_matches_black_scholes(
    option_type: OptionType,
    method: FiniteDifferenceMethod,
    n_time_steps: int,
    tolerance: float,
) -> None:
    option = EuropeanOption(option_type, strike=100.0, maturity_years=1.0)
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)
    config = FiniteDifferenceConfig(
        n_spot_steps=120,
        n_time_steps=n_time_steps,
        method=method,
    )

    result = finite_difference_price(option, market, config)
    analytical_price = black_scholes_price(option, market)

    assert isclose(result.price, analytical_price, abs_tol=tolerance)


def test_finite_difference_result_contains_plot_ready_grids() -> None:
    option = EuropeanOption(OptionType.CALL, strike=100.0, maturity_years=1.0)
    market = BlackScholesMarket(spot=100.0, rate=0.05, volatility=0.2)
    config = FiniteDifferenceConfig(
        n_spot_steps=40,
        n_time_steps=20,
        method=FiniteDifferenceMethod.CRANK_NICOLSON,
    )

    result = finite_difference_price(option, market, config)

    assert result.spot_grid.shape == (41,)
    assert result.time_to_expiry_grid.shape == (21,)
    assert result.values.shape == (21, 41)
    assert result.values[0, 0] == 0.0
    np.testing.assert_array_equal(
        result.values[0],
        np.maximum(result.spot_grid - option.strike, 0.0),
    )


def test_finite_difference_zero_maturity_returns_intrinsic_value() -> None:
    option = EuropeanOption(OptionType.PUT, strike=100.0, maturity_years=0.0)
    market = BlackScholesMarket(spot=90.0, rate=0.05, volatility=0.2)

    result = finite_difference_price(option, market)

    assert result.price == 10.0


def test_finite_difference_config_rejects_too_few_spot_steps() -> None:
    with pytest.raises(ValueError, match="n_spot_steps must be at least 3"):
        FiniteDifferenceConfig(n_spot_steps=2)


def test_finite_difference_config_rejects_non_positive_time_steps() -> None:
    with pytest.raises(ValueError, match="n_time_steps must be positive"):
        FiniteDifferenceConfig(n_time_steps=0)


def test_finite_difference_config_rejects_low_spot_max_multiplier() -> None:
    with pytest.raises(ValueError, match="spot_max_multiplier must be greater than 1"):
        FiniteDifferenceConfig(spot_max_multiplier=1.0)

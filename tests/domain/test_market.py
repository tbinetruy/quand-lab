import pytest

from quant_lab.domain import BlackScholesMarket


def test_black_scholes_market_accepts_valid_inputs() -> None:
    market = BlackScholesMarket(
        spot=100.0,
        rate=0.05,
        volatility=0.2,
        dividend_yield=0.01,
    )

    assert market.spot == 100.0
    assert market.rate == 0.05
    assert market.volatility == 0.2
    assert market.dividend_yield == 0.01


@pytest.mark.parametrize("spot", [0.0, -1.0])
def test_black_scholes_market_rejects_non_positive_spot(spot: float) -> None:
    with pytest.raises(ValueError, match="spot must be positive"):
        BlackScholesMarket(spot=spot, rate=0.05, volatility=0.2)


def test_black_scholes_market_rejects_negative_volatility() -> None:
    with pytest.raises(ValueError, match="volatility must be non-negative"):
        BlackScholesMarket(spot=100.0, rate=0.05, volatility=-0.1)


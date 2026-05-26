from dataclasses import FrozenInstanceError

import pytest

from quant_lab.domain import EuropeanOption, OptionType


def test_european_option_accepts_valid_call() -> None:
    option = EuropeanOption(
        option_type=OptionType.CALL,
        strike=100.0,
        maturity_years=1.0,
    )

    assert option.option_type is OptionType.CALL
    assert option.strike == 100.0
    assert option.maturity_years == 1.0


def test_european_option_is_immutable() -> None:
    option = EuropeanOption(
        option_type=OptionType.CALL,
        strike=100.0,
        maturity_years=1.0,
    )

    with pytest.raises(FrozenInstanceError):
        setattr(option, "strike", 101.0)  # noqa: B010


@pytest.mark.parametrize("strike", [0.0, -1.0])
def test_european_option_rejects_non_positive_strike(strike: float) -> None:
    with pytest.raises(ValueError, match="strike must be positive"):
        EuropeanOption(
            option_type=OptionType.PUT,
            strike=strike,
            maturity_years=1.0,
        )


def test_european_option_rejects_negative_maturity() -> None:
    with pytest.raises(ValueError, match="maturity_years must be non-negative"):
        EuropeanOption(
            option_type=OptionType.CALL,
            strike=100.0,
            maturity_years=-0.1,
        )

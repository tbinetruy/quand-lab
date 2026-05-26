import numpy as np

from quant_lab.domain import EuropeanOption, OptionType
from quant_lab.pricing import european_payoff


def test_european_call_payoff() -> None:
    option = EuropeanOption(OptionType.CALL, strike=100.0, maturity_years=1.0)
    terminal_prices = np.array([90.0, 100.0, 110.0], dtype=np.float64)

    payoffs = european_payoff(option, terminal_prices)

    np.testing.assert_array_equal(payoffs, np.array([0.0, 0.0, 10.0], dtype=np.float64))


def test_european_put_payoff() -> None:
    option = EuropeanOption(OptionType.PUT, strike=100.0, maturity_years=1.0)
    terminal_prices = np.array([90.0, 100.0, 110.0], dtype=np.float64)

    payoffs = european_payoff(option, terminal_prices)

    np.testing.assert_array_equal(payoffs, np.array([10.0, 0.0, 0.0], dtype=np.float64))


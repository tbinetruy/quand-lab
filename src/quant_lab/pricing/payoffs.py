from __future__ import annotations

import numpy as np

from quant_lab.domain import EuropeanOption, FloatArray, OptionType


def european_payoff(option: EuropeanOption, terminal_prices: FloatArray) -> FloatArray:
    """Evaluate a European option payoff on terminal underlying prices."""

    if option.option_type is OptionType.CALL:
        return np.maximum(terminal_prices - option.strike, 0.0)

    return np.maximum(option.strike - terminal_prices, 0.0)


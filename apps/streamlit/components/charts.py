from __future__ import annotations

import numpy as np

from quant_lab.domain import EuropeanOption, FloatArray, OptionType
from quant_lab.pricing import european_payoff

ChartRow = dict[str, float | str]
ChartSpec = dict[str, object]


def float_array(values: list[float]) -> FloatArray:
    return np.array(values, dtype=np.float64)


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count == 1:
        return [start]

    step = (stop - start) / (count - 1)
    return [start + step * index for index in range(count)]


def padded_range(values: list[float]) -> tuple[float, float]:
    lower_bound = min(values)
    upper_bound = max(values)
    span = upper_bound - lower_bound

    if span == 0.0:
        padding = max(abs(lower_bound) * 0.05, 1.0)
    else:
        padding = span * 0.1

    return lower_bound - padding, upper_bound + padding


def surface_value_min(rows: list[ChartRow], *, field: str = "price") -> float:
    return min(float(row[field]) for row in rows)


def surface_value_max(rows: list[ChartRow], *, field: str = "price") -> float:
    return max(float(row[field]) for row in rows)


def payoff_rows(
    *,
    option_type: OptionType,
    payoff_spots: list[float],
    strike: float,
    maturity: float,
) -> list[ChartRow]:
    spot_array = float_array(payoff_spots)
    option = EuropeanOption(option_type, strike=strike, maturity_years=maturity)
    payoffs = european_payoff(option, spot_array)

    return [
        {
            "spot": payoff_spot,
            "payoff": float(payoff),
            "option_type": option_type.value.title(),
        }
        for payoff_spot, payoff in zip(payoff_spots, payoffs, strict=True)
    ]


def payoff_chart_spec() -> ChartSpec:
    return {
        "mark": {"type": "line", "point": False},
        "encoding": {
            "x": {"field": "spot", "type": "quantitative", "title": "Spot at maturity"},
            "y": {"field": "payoff", "type": "quantitative", "title": "Payoff"},
            "tooltip": [
                {"field": "option_type", "type": "nominal", "title": "Option"},
                {"field": "spot", "type": "quantitative"},
                {"field": "payoff", "type": "quantitative"},
            ],
        },
    }


def line_chart_spec(
    *,
    x_field: str,
    x_title: str,
    y_field: str,
    y_title: str,
    y_min: float | None = None,
    y_max: float | None = None,
    color_field: str | None = "series",
) -> ChartSpec:
    y_encoding: ChartSpec = {
        "field": y_field,
        "type": "quantitative",
        "title": y_title,
    }
    if y_min is not None and y_max is not None:
        y_encoding["scale"] = {"domain": [y_min, y_max], "zero": False}

    encoding: ChartSpec = {
        "x": {"field": x_field, "type": "quantitative", "title": x_title},
        "y": y_encoding,
    }
    if color_field is not None:
        encoding["color"] = {"field": color_field, "type": "nominal", "title": ""}

    return {
        "mark": {"type": "line", "point": True},
        "encoding": encoding,
    }


def surface_chart_spec(
    *,
    value_title: str,
    value_min: float,
    value_max: float,
    x_field: str = "spot",
    x_title: str = "Spot",
    y_field: str = "volatility",
    y_title: str = "Volatility",
    value_field: str = "price",
) -> ChartSpec:
    return {
        "mark": "rect",
        "encoding": {
            "x": {"field": x_field, "type": "ordinal", "title": x_title, "sort": "ascending"},
            "y": {"field": y_field, "type": "ordinal", "title": y_title, "sort": "ascending"},
            "color": {
                "field": value_field,
                "type": "quantitative",
                "title": value_title,
                "scale": {"domain": [value_min, value_max]},
            },
            "tooltip": [
                {"field": "option_type", "type": "nominal", "title": "Option"},
                {"field": x_field, "type": "quantitative", "title": x_title},
                {"field": y_field, "type": "quantitative", "title": y_title},
                {"field": value_field, "type": "quantitative", "title": value_title},
            ],
        },
    }


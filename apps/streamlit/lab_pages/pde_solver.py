from __future__ import annotations

from components.charts import (
    ChartRow,
    line_chart_spec,
    padded_range,
    surface_chart_spec,
    surface_value_max,
    surface_value_min,
)
from components.source import show_source_file

import streamlit as st
from quant_lab.domain import BlackScholesMarket, EuropeanOption, OptionType
from quant_lab.pricing import (
    FiniteDifferenceConfig,
    FiniteDifferenceMethod,
    FiniteDifferenceResult,
    black_scholes_price,
    finite_difference_price,
)


def render() -> None:
    st.header("Finite-Difference PDE Solver")
    st.caption("Price European options by discretizing the Black-Scholes PDE.")

    st.markdown(
        """
        Instead of simulating paths, finite-difference solvers approximate the
        Black-Scholes PDE on a grid of stock prices and time-to-expiry values.
        The result is a full option-value surface, not just one price.
        """
    )

    st.subheader("Concept")
    st.markdown(
        """
        We solve forward in time-to-expiry $\\tau = T - t$, starting from the
        payoff at expiry and stepping toward today's value.

        Boundary conditions encode obvious behavior at the edges of the stock
        grid: calls are nearly worthless near zero spot, while puts approach the
        discounted strike near zero spot.
        """
    )

    st.subheader("Math")
    st.markdown(
        r"$$\frac{\partial V}{\partial \tau}="
        r"\frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2}"
        r"+(r-q)S\frac{\partial V}{\partial S}-rV$$"
    )
    st.markdown(r"$$V(S,0)=\Phi(S)$$")

    st.subheader("Implementation")
    show_source_file("src/quant_lab/pricing/finite_difference.py")

    st.subheader("Experiment")
    left, right = st.columns(2)

    with left:
        selected_option_type = st.segmented_control(
            "Option type",
            options=["call", "put"],
            format_func=lambda value: value.title(),
            default="call",
            key="pde_option_type",
        )
        option_type = OptionType(selected_option_type or "call")
        selected_method = st.selectbox(
            "Scheme",
            options=list(FiniteDifferenceMethod),
            format_func=lambda value: value.value.replace("_", " ").title(),
            index=2,
        )
        strike = st.number_input("Strike", min_value=0.01, value=100.0, step=1.0)
        maturity = st.number_input("Maturity in years", min_value=0.0, value=1.0, step=0.25)

    with right:
        spot = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0)
        rate = st.number_input("Risk-free rate", value=0.05, step=0.01, format="%.4f")
        dividend_yield = st.number_input("Dividend yield", value=0.0, step=0.01, format="%.4f")
        volatility = st.number_input("Volatility", min_value=0.0, value=0.2, step=0.05)
        n_spot_steps = st.number_input(
            "Spot steps",
            min_value=20,
            max_value=300,
            value=100,
            step=10,
        )
        n_time_steps = st.number_input(
            "Time steps",
            min_value=10,
            max_value=5_000,
            value=300,
            step=50,
        )

    option = EuropeanOption(option_type=option_type, strike=strike, maturity_years=maturity)
    market = BlackScholesMarket(
        spot=spot,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    config = FiniteDifferenceConfig(
        n_spot_steps=n_spot_steps,
        n_time_steps=n_time_steps,
        method=FiniteDifferenceMethod(selected_method or FiniteDifferenceMethod.CRANK_NICOLSON),
    )
    result = finite_difference_price(option, market, config)
    analytical_price = black_scholes_price(option, market)

    metric_left, metric_middle, metric_right = st.columns(3)
    metric_left.metric("PDE price", f"{result.price:.4f}")
    metric_middle.metric("Black-Scholes price", f"{analytical_price:.4f}")
    metric_right.metric("Absolute error", f"{abs(result.price - analytical_price):.4f}")

    st.json(
        {
            "scheme": result.method.value,
            "stability_ratio": round(result.stability_ratio, 6),
            "explicit_scheme_note": _stability_note(result),
        }
    )

    st.markdown(
        """
        Option value surface:

        The raw option value is dominated by intrinsic value as spot increases,
        so time effects can be visually subtle on the full stock grid.
        """
    )
    value_surface_rows = _value_surface_rows(option, result)
    st.vega_lite_chart(
        value_surface_rows,
        surface_chart_spec(
            value_title="Option value",
            value_min=surface_value_min(value_surface_rows, field="value"),
            value_max=surface_value_max(value_surface_rows, field="value"),
            y_field="time_to_expiry",
            y_title="Time to expiry",
            value_field="value",
        ),
        use_container_width=True,
    )

    st.markdown("Time value surface:")
    st.vega_lite_chart(
        value_surface_rows,
        surface_chart_spec(
            value_title="Option value - intrinsic value",
            value_min=surface_value_min(value_surface_rows, field="time_value"),
            value_max=surface_value_max(value_surface_rows, field="time_value"),
            y_field="time_to_expiry",
            y_title="Time to expiry",
            value_field="time_value",
        ),
        use_container_width=True,
    )

    st.markdown("Final price curve:")
    final_curve_rows = _final_curve_rows(option, market, result)
    final_values = [float(row["value"]) for row in final_curve_rows]
    y_min, y_max = padded_range(final_values)
    st.vega_lite_chart(
        final_curve_rows,
        line_chart_spec(
            x_field="spot",
            x_title="Spot",
            y_field="value",
            y_title="Option value",
            y_min=y_min,
            y_max=y_max,
        ),
        use_container_width=True,
    )

    st.markdown("Error against Black-Scholes:")
    error_rows = _error_rows(option, market, result)
    error_values = [float(row["error"]) for row in error_rows]
    error_min, error_max = padded_range([*error_values, 0.0])
    st.vega_lite_chart(
        error_rows,
        line_chart_spec(
            x_field="spot",
            x_title="Spot",
            y_field="error",
            y_title="PDE - Black-Scholes",
            y_min=error_min,
            y_max=error_max,
            color_field=None,
        ),
        use_container_width=True,
    )


def _stability_note(result: FiniteDifferenceResult) -> str:
    if result.method is not FiniteDifferenceMethod.EXPLICIT:
        return (
            "Implicit and Crank-Nicolson schemes are not constrained by the same "
            "explicit stability limit."
        )
    if result.stability_ratio <= 1.0:
        return "Explicit scheme is within the rough sigma^2 * M^2 * dt <= 1 stability guideline."
    return "Explicit scheme exceeds the rough sigma^2 * M^2 * dt <= 1 stability guideline."


def _value_surface_rows(
    option: EuropeanOption,
    result: FiniteDifferenceResult,
) -> list[ChartRow]:
    rows: list[ChartRow] = []
    spot_stride = max(1, len(result.spot_grid) // 60)
    time_stride = max(1, len(result.time_to_expiry_grid) // 60)

    for time_index in range(0, len(result.time_to_expiry_grid), time_stride):
        for spot_index in range(0, len(result.spot_grid), spot_stride):
            spot = float(result.spot_grid[spot_index])
            value = float(result.values[time_index, spot_index])
            intrinsic_value = _intrinsic_value(option, spot)
            rows.append(
                {
                    "spot": round(spot, 2),
                    "time_to_expiry": round(float(result.time_to_expiry_grid[time_index]), 4),
                    "value": round(value, 6),
                    "time_value": round(value - intrinsic_value, 6),
                    "option_type": option.option_type.value.title(),
                }
            )

    return rows


def _final_curve_rows(
    option: EuropeanOption,
    market: BlackScholesMarket,
    result: FiniteDifferenceResult,
) -> list[ChartRow]:
    rows: list[ChartRow] = []
    for spot, pde_value in zip(result.spot_grid, result.values[-1], strict=True):
        safe_spot = max(float(spot), 1e-12)
        analytical_market = _replace_market(market, spot=safe_spot)
        rows.append(
            {
                "spot": round(float(spot), 4),
                "series": "PDE",
                "value": float(pde_value),
                "option_type": option.option_type.value.title(),
            }
        )
        rows.append(
            {
                "spot": round(float(spot), 4),
                "series": "Black-Scholes",
                "value": black_scholes_price(option, analytical_market),
                "option_type": option.option_type.value.title(),
            }
        )

    return rows


def _error_rows(
    option: EuropeanOption,
    market: BlackScholesMarket,
    result: FiniteDifferenceResult,
) -> list[ChartRow]:
    rows: list[ChartRow] = []
    for spot, pde_value in zip(result.spot_grid, result.values[-1], strict=True):
        safe_spot = max(float(spot), 1e-12)
        analytical_market = _replace_market(market, spot=safe_spot)
        analytical_value = black_scholes_price(option, analytical_market)
        rows.append(
            {
                "spot": round(float(spot), 4),
                "error": float(pde_value) - analytical_value,
                "option_type": option.option_type.value.title(),
            }
        )

    return rows


def _intrinsic_value(option: EuropeanOption, spot: float) -> float:
    if option.option_type is OptionType.CALL:
        return max(spot - option.strike, 0.0)
    return max(option.strike - spot, 0.0)


def _replace_market(market: BlackScholesMarket, *, spot: float) -> BlackScholesMarket:
    return BlackScholesMarket(
        spot=spot,
        rate=market.rate,
        volatility=market.volatility,
        dividend_yield=market.dividend_yield,
    )

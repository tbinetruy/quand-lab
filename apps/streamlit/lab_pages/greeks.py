from __future__ import annotations

from components.charts import (
    ChartRow,
    line_chart_spec,
    linspace,
    padded_range,
    surface_chart_spec,
)
from components.source import show_source_file

import streamlit as st
from quant_lab.domain import BlackScholesMarket, EuropeanOption, OptionType
from quant_lab.risk import black_scholes_greeks, finite_difference_greek
from quant_lab.risk.greeks import BlackScholesGreeks, GreekName

GREEK_NAMES: list[GreekName] = ["delta", "gamma", "vega", "theta", "rho"]


def render() -> None:
    st.header("Greeks")
    st.caption("Sensitivities of the Black-Scholes price to market and contract inputs.")

    st.markdown(
        """
        Greeks measure how an option price changes when one input changes. They
        are local sensitivities: useful for understanding risk near the current
        market state, not global guarantees about every possible scenario.
        """
    )

    st.subheader("Concept")
    st.markdown(
        """
        The Black-Scholes price is one number, but a trading book needs to know
        how that number changes when the market moves. Greeks are local
        sensitivities of the price function $V$.

        Local means "near the current inputs." A delta of $0.55$ does not mean
        the option will behave like $0.55$ shares forever. It means that for a
        small spot move around the current market state, the option price should
        move roughly like $0.55$ shares.
        """
    )
    st.table(
        [
            {
                "Greek": "Delta",
                "Practical question": "How much stock exposure does the option behave like?",
                "Common use": "Directional exposure and hedge ratio.",
            },
            {
                "Greek": "Gamma",
                "Practical question": "How quickly will delta change if spot moves?",
                "Common use": "Delta hedge stability and convexity risk.",
            },
            {
                "Greek": "Vega",
                "Practical question": "How much does the option care about volatility?",
                "Common use": "Volatility exposure and implied-vol scenarios.",
            },
            {
                "Greek": "Theta",
                "Practical question": "How does value change as time passes?",
                "Common use": "Time decay and carry analysis.",
            },
            {
                "Greek": "Rho",
                "Practical question": "How sensitive is the price to interest rates?",
                "Common use": "Rate exposure, especially for longer maturities.",
            },
        ]
    )
    st.markdown(
        """
        Greeks are useful because they turn a nonlinear payoff into a set of
        local risk numbers that can be aggregated, hedged, and stress tested.
        They are not a complete risk model: large moves, changing volatility,
        jumps, liquidity, and model error can all make local sensitivities
        misleading.
        """
    )

    st.subheader("Why This Matters Later")
    st.markdown(
        """
        Greeks are the bridge from pricing one option to managing a book of
        risk. Later, when we add market data and repeated model runs, Greeks
        give us a compact way to ask better questions than "what is the price?"

        They will be useful for:

        - hedging: delta tells us how much underlying offsets small spot moves
        - hedge maintenance: gamma tells us when a delta hedge will become stale
        - volatility views: vega tells us whether a position benefits from
          higher or lower implied volatility
        - carry analysis: theta tells us what happens if the market does not
          move but time passes
        - scenario analysis: Greeks give first-order and second-order estimates
          before we run full repricing shocks
        - portfolio aggregation: exposures from many options can be summed by
          underlying, maturity, volatility bucket, or strategy
        - model diagnostics: unusual Greeks can reveal unstable inputs,
          near-expiry behavior, or numerical issues
        """
    )
    st.markdown(
        """
        They can also feed portfolio construction, but usually as risk
        constraints rather than as the whole objective. For example, a strategy
        might target a volatility exposure while keeping delta near zero, or
        compare expected return against gamma and vega risk. Full optimization
        still needs forecasts, costs, liquidity, and risk limits; Greeks tell us
        what risks the candidate portfolio is carrying.
        """
    )
    st.markdown(
        """
        Symbols used below:

        - $\\Delta$: sensitivity to spot
        - $\\Gamma$: sensitivity of delta to spot
        - $\\nu$: sensitivity to volatility, commonly called vega
        - $\\Theta$: sensitivity to time passing
        - $\\rho$: sensitivity to the risk-free rate
        """
    )

    st.subheader("Math")
    st.markdown(r"$$\Delta = \frac{\partial V}{\partial S}$$")
    st.markdown(r"$$\Gamma = \frac{\partial^2 V}{\partial S^2}$$")
    st.markdown(r"$$\nu = \frac{\partial V}{\partial \sigma}$$")
    st.markdown(r"$$\Theta = \frac{\partial V}{\partial t}$$")
    st.markdown(r"$$\rho = \frac{\partial V}{\partial r}$$")
    st.markdown(
        """
        Analytical Greeks differentiate the Black-Scholes formula directly.
        Finite-difference Greeks bump an input, reprice the option, and estimate
        the derivative numerically. Comparing both is a good implementation
        check and also shows what "sensitivity" means operationally.
        """
    )

    st.subheader("Implementation")
    show_source_file("src/quant_lab/risk/greeks.py")

    st.subheader("Experiment")
    left, right = st.columns(2)

    with left:
        selected_option_type = st.segmented_control(
            "Option type",
            options=["call", "put"],
            format_func=lambda value: value.title(),
            default="call",
            key="greeks_option_type",
        )
        option_type = OptionType(selected_option_type or "call")
        selected_greek = st.selectbox(
            "Greek",
            options=GREEK_NAMES,
            format_func=lambda value: value.title(),
            index=0,
        )
        strike = st.number_input("Strike", min_value=0.01, value=100.0, step=1.0)
        maturity = st.number_input("Maturity in years", min_value=0.01, value=1.0, step=0.25)

    with right:
        spot = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0)
        rate = st.number_input("Risk-free rate", value=0.05, step=0.01, format="%.4f")
        dividend_yield = st.number_input("Dividend yield", value=0.0, step=0.01, format="%.4f")
        volatility = st.number_input("Volatility", min_value=0.01, value=0.2, step=0.05)

    option = EuropeanOption(option_type=option_type, strike=strike, maturity_years=maturity)
    market = BlackScholesMarket(
        spot=spot,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )

    greeks = black_scholes_greeks(option, market)
    finite_difference = finite_difference_greek(option, market, selected_greek)
    analytical_value = _greek_value(greeks, selected_greek)

    metric_left, metric_middle, metric_right = st.columns(3)
    metric_left.metric(f"Analytical {selected_greek.title()}", f"{analytical_value:.6f}")
    metric_middle.metric("Finite difference", f"{finite_difference:.6f}")
    metric_right.metric("Absolute error", f"{abs(analytical_value - finite_difference):.6f}")

    st.markdown(f"{selected_greek.title()} over spot:")
    st.markdown(_greek_curve_explanation(selected_greek))
    curve_rows = _greek_curve_rows(
        option=option,
        market=market,
        greek=selected_greek,
    )
    y_min, y_max = _greek_display_range(
        selected_greek,
        [float(row["value"]) for row in curve_rows],
    )
    st.vega_lite_chart(
        curve_rows,
        line_chart_spec(
            x_field="spot",
            x_title="Spot",
            y_field="value",
            y_title=selected_greek.title(),
            y_min=y_min,
            y_max=y_max,
            color_field=None,
        ),
        use_container_width=True,
    )

    st.markdown(f"{selected_greek.title()} surface over spot and volatility:")
    st.markdown(_greek_surface_explanation(selected_greek))
    surface_rows = _greek_surface_rows(
        option=option,
        market=market,
        greek=selected_greek,
    )
    surface_value_minimum, surface_value_maximum = _greek_display_range(
        selected_greek,
        [float(row["value"]) for row in surface_rows],
    )
    st.vega_lite_chart(
        surface_rows,
        surface_chart_spec(
            value_title=selected_greek.title(),
            value_min=surface_value_minimum,
            value_max=surface_value_maximum,
            value_field="value",
        ),
        use_container_width=True,
    )


def _greek_value(greeks: BlackScholesGreeks, greek: GreekName) -> float:
    if greek == "delta":
        return greeks.delta
    if greek == "gamma":
        return greeks.gamma
    if greek == "vega":
        return greeks.vega
    if greek == "theta":
        return greeks.theta
    if greek == "rho":
        return greeks.rho

    raise ValueError(f"unsupported greek: {greek}")


def _greek_display_range(greek: GreekName, values: list[float]) -> tuple[float, float]:
    if greek == "delta":
        return -1.0, 1.0
    if greek == "rho":
        limit = max(abs(value) for value in values)
        return -limit, limit

    return padded_range(values)


def _greek_curve_rows(
    *,
    option: EuropeanOption,
    market: BlackScholesMarket,
    greek: GreekName,
) -> list[ChartRow]:
    spot_min = max(0.01, market.spot * 0.5)
    spot_max = market.spot * 1.5

    rows: list[ChartRow] = []
    for curve_spot in linspace(spot_min, spot_max, 80):
        curve_market = _replace_market(market, spot=curve_spot)
        greeks = black_scholes_greeks(option, curve_market)
        rows.append(
            {
                "spot": round(curve_spot, 4),
                "value": _greek_value(greeks, greek),
                "option_type": option.option_type.value.title(),
            }
        )

    return rows


def _greek_surface_rows(
    *,
    option: EuropeanOption,
    market: BlackScholesMarket,
    greek: GreekName,
) -> list[ChartRow]:
    spot_min = max(0.01, market.spot * 0.6)
    spot_max = market.spot * 1.4
    volatility_max = max(0.6, market.volatility * 2.0, 0.1)

    rows: list[ChartRow] = []
    for surface_spot in linspace(spot_min, spot_max, 18):
        for surface_volatility in linspace(0.01, volatility_max, 18):
            surface_market = _replace_market(
                market,
                spot=surface_spot,
                volatility=surface_volatility,
            )
            greeks = black_scholes_greeks(option, surface_market)
            rows.append(
                {
                    "spot": round(surface_spot, 2),
                    "volatility": round(surface_volatility, 3),
                    "value": _greek_value(greeks, greek),
                    "option_type": option.option_type.value.title(),
                }
            )

    return rows


def _greek_curve_explanation(greek: GreekName) -> str:
    if greek == "delta":
        return (
            "Delta shows directional stock exposure. Calls move from near 0 to near 1 "
            "as spot rises through the strike. Puts move from near -1 to near 0."
        )
    if greek == "gamma":
        return (
            "Gamma is highest near the strike because that is where a small spot move "
            "most changes the option's moneyness and therefore its delta."
        )
    if greek == "vega":
        return (
            "Vega is usually largest near the strike: volatility matters most when the "
            "future payoff is uncertain."
        )
    if greek == "theta":
        return (
            "Theta shows time decay using the market convention dV/dt as calendar time "
            "moves forward. Long options often lose time value as expiry approaches."
        )
    return (
        "Rho shows rate exposure. It is often modest for short maturities and more "
        "important when cashflows are farther in the future."
    )


def _greek_surface_explanation(greek: GreekName) -> str:
    if greek == "delta":
        return (
            "The delta surface shows how hedge ratio changes across moneyness and "
            "volatility. Higher volatility smooths the transition around the strike."
        )
    if greek == "gamma":
        return (
            "The gamma surface highlights where delta hedges are unstable. Gamma tends "
            "to concentrate near the strike and can become sharper at lower volatility."
        )
    if greek == "vega":
        return (
            "The vega surface shows volatility exposure. It is concentrated where extra "
            "dispersion can most change the chance of finishing in-the-money."
        )
    if greek == "theta":
        return (
            "The theta surface shows carry across spot and volatility. Its sign and "
            "shape depend on option type, rates, dividends, and moneyness."
        )
    return (
        "The rho surface shows how rate sensitivity varies across moneyness and "
        "volatility. Longer-dated discounted strike exposure usually drives rho."
    )


def _replace_market(
    market: BlackScholesMarket,
    *,
    spot: float | None = None,
    volatility: float | None = None,
) -> BlackScholesMarket:
    return BlackScholesMarket(
        spot=market.spot if spot is None else spot,
        rate=market.rate,
        volatility=market.volatility if volatility is None else volatility,
        dividend_yield=market.dividend_yield,
    )

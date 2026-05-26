from __future__ import annotations

from math import exp

from components.charts import (
    ChartRow,
    linspace,
    payoff_chart_spec,
    payoff_rows,
    surface_chart_spec,
    surface_value_max,
    surface_value_min,
)
from components.source import show_source_file

import streamlit as st
from quant_lab.domain import BlackScholesMarket, EuropeanOption, OptionType
from quant_lab.pricing import black_scholes_price


def render() -> None:
    st.header("Black-Scholes Closed Form")
    st.caption("A first analytical benchmark for European option prices.")

    st.markdown(
        """
        Black-Scholes gives a closed-form price for European calls and puts
        under a specific set of assumptions. This is the benchmark we will use
        when we later check Monte Carlo convergence and finite-difference
        solvers.
        """
    )

    st.subheader("Concept")
    st.markdown(
        """
        We assume the underlying follows geometric Brownian motion with constant
        volatility and constant continuously compounded rates.

        Symbols used below:

        - $S_0$: current spot price
        - $K$: option strike
        - $T$: maturity in years
        - $r$: continuously compounded risk-free rate
        - $q$: continuously compounded dividend yield
        - $\\sigma$: annualized volatility
        - $N(x)$: cumulative distribution function of the standard normal distribution
        """
    )

    st.subheader("Math")
    st.markdown(
        r"""
        $$d_1 = \frac{\ln(S_0/K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}$$

        $$d_2 = d_1 - \sigma\sqrt{T}$$

        $$C = S_0 e^{-qT}N(d_1) - K e^{-rT}N(d_2)$$

        $$P = K e^{-rT}N(-d_2) - S_0 e^{-qT}N(-d_1)$$
        """
    )

    st.subheader("Implementation")
    show_source_file("src/quant_lab/pricing/closed_form.py")

    st.subheader("Experiment")
    left, right = st.columns(2)

    with left:
        selected_option_type = st.segmented_control(
            "Option type",
            options=["call", "put"],
            format_func=lambda value: value.title(),
            default="call",
            key="black_scholes_option_type",
        )
        option_type = OptionType(selected_option_type or "call")
        strike = st.number_input("Strike", min_value=0.01, value=100.0, step=1.0)
        maturity = st.number_input("Maturity in years", min_value=0.0, value=1.0, step=0.25)

    with right:
        spot = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0)
        rate = st.number_input("Risk-free rate", value=0.05, step=0.01, format="%.4f")
        dividend_yield = st.number_input("Dividend yield", value=0.0, step=0.01, format="%.4f")
        volatility = st.number_input("Volatility", min_value=0.0, value=0.2, step=0.05)

    option = EuropeanOption(
        option_type=option_type,
        strike=strike,
        maturity_years=maturity,
    )
    market = BlackScholesMarket(
        spot=spot,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )

    price = black_scholes_price(option, market)
    st.metric("Closed-form price", f"{price:.4f}")

    call = EuropeanOption(OptionType.CALL, strike=strike, maturity_years=maturity)
    put = EuropeanOption(OptionType.PUT, strike=strike, maturity_years=maturity)
    call_price = black_scholes_price(call, market)
    put_price = black_scholes_price(put, market)
    parity_left = call_price - put_price
    parity_right = spot * exp(-dividend_yield * maturity) - strike * exp(-rate * maturity)

    st.markdown("Put-call parity check:")
    st.json(
        {
            "call_price_minus_put_price": round(parity_left, 8),
            "discounted_spot_minus_discounted_strike": round(parity_right, 8),
            "absolute_error": round(abs(parity_left - parity_right), 12),
        }
    )

    st.markdown("Payoff diagram:")
    payoff_spots = linspace(max(0.01, strike * 0.5), strike * 1.5, 80)
    st.vega_lite_chart(
        payoff_rows(
            option_type=option_type,
            payoff_spots=payoff_spots,
            strike=strike,
            maturity=maturity,
        ),
        payoff_chart_spec(),
        key=f"black-scholes-payoff-{option_type.value}",
        use_container_width=True,
    )

    st.markdown("Closed-form price surface over spot and volatility:")
    surface_rows = _closed_form_surface_rows(option, market)
    st.vega_lite_chart(
        surface_rows,
        surface_chart_spec(
            value_title="Closed-form price",
            value_min=surface_value_min(surface_rows),
            value_max=surface_value_max(surface_rows),
        ),
        key=f"black-scholes-surface-{option_type.value}",
        use_container_width=True,
    )


def _closed_form_surface_rows(
    option: EuropeanOption,
    market: BlackScholesMarket,
) -> list[ChartRow]:
    spot_min = max(0.01, market.spot * 0.6)
    spot_max = market.spot * 1.4
    volatility_max = max(0.6, market.volatility * 2.0, 0.1)

    rows: list[ChartRow] = []
    for surface_spot in linspace(spot_min, spot_max, 18):
        for surface_volatility in linspace(0.01, volatility_max, 18):
            surface_market = BlackScholesMarket(
                spot=surface_spot,
                rate=market.rate,
                volatility=surface_volatility,
                dividend_yield=market.dividend_yield,
            )
            rows.append(
                {
                    "spot": round(surface_spot, 2),
                    "volatility": round(surface_volatility, 3),
                    "price": round(black_scholes_price(option, surface_market), 6),
                    "option_type": option.option_type.value.title(),
                }
            )

    return rows

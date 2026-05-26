from __future__ import annotations

from math import exp

from components.charts import (
    ChartRow,
    line_chart_spec,
    linspace,
    padded_range,
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
        solvers. If calls, puts, moneyness, or time value are new, start with
        the Foundations page first. If Ito's lemma is new, read the Stochastic
        Calculus page before this derivation.
        """
    )

    st.subheader("Concept")
    st.markdown(
        """
        Black-Scholes starts with a model for the underlying asset and derives
        the option price from a hedging argument. The key idea is no-arbitrage:
        if two portfolios have the same future cashflows, they should have the
        same price today.

        Main assumptions:

        - the underlying follows geometric Brownian motion
        - volatility is constant
        - the risk-free rate and dividend yield are constant
        - trading is continuous and frictionless
        - the option is European, so exercise happens only at maturity
        - the underlying can be bought, sold, and used for hedging

        Symbols used below:

        - $S_0$: current spot price
        - $S_t$: underlying price at time $t$
        - $K$: option strike
        - $T$: maturity in years
        - $V(S,t)$: option value when spot is $S$ at time $t$
        - $r$: continuously compounded risk-free rate
        - $q$: continuously compounded dividend yield
        - $\\sigma$: annualized volatility
        - $\\Delta$: hedge ratio, later called delta
        - $N(x)$: cumulative distribution function of the standard normal distribution
        """
    )

    st.subheader("Derivation")
    st.markdown(
        """
        Under the physical model, GBM uses an expected return $\\mu$:
        """
    )
    st.markdown(r"$$dS_t=\mu S_t\,dt+\sigma S_t\,dW_t$$")
    st.markdown(
        """
        The option value is a function of spot and time: $V(S,t)$. Ito's lemma
        is the stochastic-calculus rule for differentiating such a function when
        $S_t$ has Brownian noise. At a high level, it says the option value
        changes because time passes, spot moves, and Brownian motion has
        quadratic variation:
        """
    )
    st.markdown(
        "$$"
        r"dV = \left(\frac{\partial V}{\partial t}"
        r"+\mu S\frac{\partial V}{\partial S}"
        r"+\frac{1}{2}\sigma^2S^2\frac{\partial^2V}{\partial S^2}\right)dt"
        r"+\sigma S\frac{\partial V}{\partial S}dW_t"
        "$$"
    )
    st.markdown(
        """
        The random term is the part multiplied by $dW_t$. Black-Scholes removes
        that randomness by holding one option and shorting $\\Delta$ shares of
        the underlying:
        """
    )
    st.markdown(r"$$\Pi = V - \Delta S$$")
    st.markdown(
        """
        Choose $\\Delta = \\frac{\\partial V}{\\partial S}$. The Brownian shock in
        the option is then offset by the Brownian shock in the stock hedge. The
        hedged portfolio is locally riskless, so by no-arbitrage it must earn
        the risk-free rate. That argument gives the Black-Scholes PDE:
        """
    )
    st.markdown(
        "$$"
        r"\frac{\partial V}{\partial t}"
        r"+\frac{1}{2}\sigma^2S^2\frac{\partial^2 V}{\partial S^2}"
        r"+(r-q)S\frac{\partial V}{\partial S}"
        r"-rV=0"
        "$$"
    )
    st.markdown(
        """
        Notice that $\\mu$ disappeared. The option price does not use the
        investor's expected stock return directly; the hedge removes that source
        of risk from the pricing equation. This is one way to understand why
        risk-neutral pricing works.
        """
    )

    st.subheader("Closed Form")
    st.markdown(
        """
        For European calls and puts, the PDE can be transformed into the heat
        equation and solved analytically. We will not do the full heat-equation
        transformation here; the important outcome is that the terminal payoff
        plus GBM lognormality leads to the formulas below.

        The quantities $d_1$ and $d_2$ are standardized log-moneyness terms.
        Roughly, $d_2$ is tied to the probability of finishing in-the-money
        under the risk-neutral distribution, while $d_1$ appears in the
        stock-weighted part of the expected payoff.
        """
    )
    st.markdown(
        r"""
        $$d_1 = \frac{\ln(S_0/K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}$$

        $$d_2 = d_1 - \sigma\sqrt{T}$$

        $$C = S_0 e^{-qT}N(d_1) - K e^{-rT}N(d_2)$$

        $$P = K e^{-rT}N(-d_2) - S_0 e^{-qT}N(-d_1)$$
        """
    )
    st.markdown(
        """
        Put-call parity is a useful consistency check. A call minus a put with
        the same strike and maturity has the same payoff as a forward-like
        position, so the prices must satisfy:
        """
    )
    st.markdown(r"$$C-P=S_0e^{-qT}-Ke^{-rT}$$")

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

    st.markdown(
        """
        Payoff diagram:

        This is the terminal cashflow, not today's model price. The call payoff
        bends upward above the strike. The put payoff bends upward below the
        strike.
        """
    )
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

    st.markdown(
        """
        Price curve:

        The model price is smoother than the payoff because it includes time
        value. Around the strike, uncertainty about finishing in-the-money is
        most important, so the curve rounds off the payoff kink.
        """
    )
    price_curve_rows = _price_curve_rows(option, market)
    curve_min, curve_max = padded_range([float(row["price"]) for row in price_curve_rows])
    st.vega_lite_chart(
        price_curve_rows,
        line_chart_spec(
            x_field="spot",
            x_title="Spot",
            y_field="price",
            y_title="Closed-form price",
            y_min=curve_min,
            y_max=curve_max,
            color_field=None,
        ),
        key=f"black-scholes-price-curve-{option_type.value}",
        use_container_width=True,
    )

    st.markdown(
        """
        Closed-form price surface over spot and volatility:

        Calls generally become more valuable as spot rises; puts generally
        become more valuable as spot falls. Both calls and puts usually become
        more valuable as volatility rises because wider future outcomes increase
        the value of optionality.
        """
    )
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


def _price_curve_rows(
    option: EuropeanOption,
    market: BlackScholesMarket,
) -> list[ChartRow]:
    spot_min = max(0.01, option.strike * 0.5)
    spot_max = option.strike * 1.5

    rows: list[ChartRow] = []
    for curve_spot in linspace(spot_min, spot_max, 120):
        curve_market = BlackScholesMarket(
            spot=curve_spot,
            rate=market.rate,
            volatility=market.volatility,
            dividend_yield=market.dividend_yield,
        )
        rows.append(
            {
                "spot": round(curve_spot, 4),
                "price": black_scholes_price(option, curve_market),
                "option_type": option.option_type.value.title(),
            }
        )

    return rows


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

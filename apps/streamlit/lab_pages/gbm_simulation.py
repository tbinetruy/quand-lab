from __future__ import annotations

from math import exp

from components.source import show_source_file

import streamlit as st
from quant_lab.domain import BlackScholesMarket, MonteCarloConfig
from quant_lab.stochastic import simulate_geometric_brownian_motion


def render() -> None:
    st.header("Geometric Brownian Motion")
    st.caption("The risk-neutral stock process behind Black-Scholes simulation.")

    st.markdown(
        """
        Geometric Brownian motion is the standard path model used in the
        Black-Scholes world. It keeps prices positive by modeling proportional
        returns instead of absolute price changes.
        """
    )

    st.subheader("Concept")
    st.markdown(
        """
        Under the risk-neutral measure, the expected growth rate of the
        underlying is the risk-free rate minus dividend yield. The random part
        is controlled by volatility and Brownian shocks.

        Symbols used below:

        - $S_t$: underlying price at time $t$
        - $r$: continuously compounded risk-free rate
        - $q$: continuously compounded dividend yield
        - $\\sigma$: annualized volatility
        - $\\Delta t$: one simulation time step
        - $Z$: a standard normal random variable
        """
    )

    st.subheader("Math")
    st.markdown(
        "$$"
        r"S_{t+\Delta t} = S_t \exp\left("
        r"(r-q-\frac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}Z"
        r"\right)"
        "$$"
    )
    st.markdown(
        r"""
        $$\operatorname{E}^{Q}[S_T] = S_0 e^{(r-q)T}$$
        """
    )

    st.subheader("Implementation")
    brownian_tab, gbm_tab = st.tabs(["Brownian increments", "GBM paths"])

    with brownian_tab:
        show_source_file("src/quant_lab/stochastic/brownian.py")

    with gbm_tab:
        show_source_file("src/quant_lab/stochastic/gbm.py")

    st.subheader("Experiment")
    left, right = st.columns(2)

    with left:
        spot = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0)
        maturity = st.number_input("Maturity in years", min_value=0.0, value=1.0, step=0.25)
        rate = st.number_input("Risk-free rate", value=0.05, step=0.01, format="%.4f")
        dividend_yield = st.number_input("Dividend yield", value=0.0, step=0.01, format="%.4f")

    with right:
        volatility = st.number_input("Volatility", min_value=0.0, value=0.2, step=0.05)
        n_paths = st.number_input("Paths", min_value=1, max_value=20_000, value=2_000, step=500)
        n_steps = st.number_input("Time steps", min_value=1, max_value=1_000, value=252, step=21)
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
        antithetic = st.toggle("Antithetic variates", value=False)

    market = BlackScholesMarket(
        spot=spot,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    config = MonteCarloConfig(
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        antithetic=antithetic,
    )
    paths = simulate_geometric_brownian_motion(market, maturity, config)
    terminal_prices = paths[:, -1]
    expected_terminal_mean = spot * exp((rate - dividend_yield) * maturity)

    st.markdown("Sample paths:")
    sample_path_count = min(25, n_paths)
    st.line_chart(paths[:sample_path_count].T)

    st.markdown("Terminal price summary:")
    st.json(
        {
            "simulated_terminal_mean": round(float(terminal_prices.mean()), 6),
            "risk_neutral_expected_terminal_mean": round(expected_terminal_mean, 6),
            "terminal_standard_deviation": round(float(terminal_prices.std(ddof=1)), 6),
            "min_terminal_price": round(float(terminal_prices.min()), 6),
            "max_terminal_price": round(float(terminal_prices.max()), 6),
        }
    )

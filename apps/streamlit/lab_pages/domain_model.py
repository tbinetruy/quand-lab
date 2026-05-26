from __future__ import annotations

from components.source import show_source_file

import streamlit as st
from quant_lab.domain import BlackScholesMarket, EuropeanOption, MonteCarloConfig, OptionType


def render() -> None:
    st.header("Domain Model")
    st.caption("The small vocabulary that later pricing models will use.")

    st.markdown(
        """
        A pricing engine needs a few stable concepts before it needs formulas.
        We start by separating the contract, the market inputs, and the
        simulation settings. That keeps the engine reusable from Streamlit,
        tests, scripts, scheduled jobs, or a future Django application.
        """
    )

    st.subheader("Concept")
    st.markdown(
        """
        The first instrument is a European option: a call or put with a strike
        and a maturity. The Black-Scholes market state is separate: spot,
        continuously compounded rate, volatility, and optional dividend yield.

        Monte Carlo settings are also separate from market data. Number of
        paths, time steps, seeds, and output choices describe how we run an
        experiment, not what the market is.
        """
    )
    st.markdown(
        """
        We will use these symbols throughout the first pricing labs:

        - $S_0$: the current spot price of the underlying asset
        - $K$: the option strike
        - $T$: the maturity in years
        - $r$: the continuously compounded risk-free rate
        - $q$: the continuously compounded dividend yield
        - $\\sigma$: the annualized volatility
        - $N$: the number of Monte Carlo paths
        - $M$: the number of time steps per path
        """
    )

    st.subheader("Math")
    st.markdown(
        r"""
        $$S_0 > 0,\quad K > 0,\quad T \ge 0,\quad \sigma \ge 0,\quad N > 0,\quad M > 0$$
        """
    )
    st.markdown(
        """
        These are not model equations yet. They are basic validity constraints.
        Encoding them at the boundary keeps later formulas focused on finance
        instead of defensive checks.
        """
    )

    st.subheader("Implementation")
    source_tab, market_tab, simulation_tab = st.tabs(["Instruments", "Market", "Simulation"])

    with source_tab:
        show_source_file("src/quant_lab/domain/instruments.py")

    with market_tab:
        show_source_file("src/quant_lab/domain/market.py")

    with simulation_tab:
        show_source_file("src/quant_lab/domain/simulation.py")

    st.subheader("Experiment")
    left, right = st.columns(2)

    with left:
        selected_option_type = st.segmented_control(
            "Option type",
            options=[OptionType.CALL, OptionType.PUT],
            format_func=lambda value: value.value.title(),
            default=OptionType.CALL,
        )
        option_type = selected_option_type or OptionType.CALL
        strike = st.number_input("Strike", min_value=0.01, value=100.0, step=1.0)
        maturity = st.number_input(
            "Maturity in years",
            min_value=0.0,
            value=1.0,
            step=0.25,
        )

    with right:
        spot = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0)
        rate = st.number_input("Risk-free rate", value=0.05, step=0.01, format="%.4f")
        volatility = st.number_input("Volatility", min_value=0.0, value=0.2, step=0.05)
        n_paths = st.number_input("Monte Carlo paths", min_value=1, value=10_000, step=1_000)

    option = EuropeanOption(
        option_type=option_type,
        strike=strike,
        maturity_years=maturity,
    )
    market = BlackScholesMarket(
        spot=spot,
        rate=rate,
        volatility=volatility,
    )
    config = MonteCarloConfig(n_paths=n_paths, n_steps=252, seed=42)

    st.markdown("The controls above create these immutable engine objects:")
    st.json(
        {
            "option": {
                "option_type": option.option_type.value,
                "strike": option.strike,
                "maturity_years": option.maturity_years,
            },
            "market": {
                "spot": market.spot,
                "rate": market.rate,
                "volatility": market.volatility,
                "dividend_yield": market.dividend_yield,
            },
            "monte_carlo_config": {
                "n_paths": config.n_paths,
                "n_steps": config.n_steps,
                "seed": config.seed,
                "antithetic": config.antithetic,
                "return_paths": config.return_paths,
            },
        }
    )


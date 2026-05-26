from __future__ import annotations

from lab_pages import (
    applied_risk,
    black_scholes,
    domain_model,
    formal_katas,
    foundations,
    gbm_simulation,
    greeks,
    monte_carlo_pricing,
    pde_solver,
    probability_primer,
    stochastic_calculus,
)

import streamlit as st
from quant_lab import __version__

INTRO_TRACK = "Quant Foundations: Introduction and Intuition"
FORMAL_TRACK = "Formal Katas: Rebuilding the Machinery"

INTRO_PAGES = [
    "Introduction",
    "Foundations",
    "Probability Primer",
    "Stochastic Calculus",
    "Domain Model",
    "GBM Simulation",
    "Black-Scholes",
    "Monte Carlo Pricing",
    "Greeks",
    "PDE Solver",
    "Applied Risk",
]

FORMAL_PAGES = [
    "Formal Katas Overview",
]


def render_intro() -> None:
    st.header("Quant Lab")
    st.caption(f"Version {__version__}")

    st.markdown(
        """
        Quant Lab is a guided quantitative finance playground. Each lab follows
        the same loop: introduce the concept, state the math, inspect the
        implementation, and then experiment with parameters.
        """
    )

    st.subheader("Current Path")
    st.markdown(
        """
        1. Build a small domain vocabulary.
        2. Learn the basic option payoff and pricing vocabulary.
        3. Introduce the probability tools behind stochastic models.
        4. Introduce stochastic calculus tools.
        5. Simulate geometric Brownian motion.
        6. Implement Black-Scholes closed-form pricing.
        7. Compare Monte Carlo prices against the analytical result.
        8. Add Greeks, hedging, and finite-difference solvers.
        9. Apply the tools to option strategies and portfolio risk.
        """
    )

    st.subheader("Architecture Rule")
    st.markdown(
        """
        The quant engine lives under `src/quant_lab` and does not import
        Streamlit. This app is only an interface over the same typed Python code
        that tests, scripts, scheduled jobs, or a future Django app can call.
        """
    )


def main() -> None:
    st.set_page_config(page_title="Quant Lab", page_icon="QL", layout="wide")

    with st.sidebar:
        st.title("Quant Lab")
        track = st.radio(
            "Learning track",
            options=[
                INTRO_TRACK,
                FORMAL_TRACK,
            ],
        )
        st.divider()
        page_options = FORMAL_PAGES if track == FORMAL_TRACK else INTRO_PAGES
        page = st.radio(
            "Page",
            options=page_options,
        )

    if page == "Formal Katas Overview":
        formal_katas.render()
    elif page == "Applied Risk":
        applied_risk.render()
    elif page == "PDE Solver":
        pde_solver.render()
    elif page == "Greeks":
        greeks.render()
    elif page == "Monte Carlo Pricing":
        monte_carlo_pricing.render()
    elif page == "GBM Simulation":
        gbm_simulation.render()
    elif page == "Black-Scholes":
        black_scholes.render()
    elif page == "Domain Model":
        domain_model.render()
    elif page == "Stochastic Calculus":
        stochastic_calculus.render()
    elif page == "Probability Primer":
        probability_primer.render()
    elif page == "Foundations":
        foundations.render()
    else:
        render_intro()


if __name__ == "__main__":
    main()

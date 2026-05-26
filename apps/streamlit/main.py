from __future__ import annotations

from lab_pages import domain_model

import streamlit as st
from quant_lab import __version__


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
        2. Implement Black-Scholes closed-form pricing.
        3. Simulate geometric Brownian motion.
        4. Compare Monte Carlo prices against the analytical result.
        5. Add Greeks, hedging, and finite-difference solvers.
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
        page = st.radio(
            "Lab",
            options=["Introduction", "Domain Model"],
            label_visibility="collapsed",
        )

    if page == "Domain Model":
        domain_model.render()
    else:
        render_intro()


if __name__ == "__main__":
    main()

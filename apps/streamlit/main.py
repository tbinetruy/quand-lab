from __future__ import annotations

import streamlit as st
from quant_lab import __version__


def main() -> None:
    st.set_page_config(page_title="Quant Lab", page_icon="QL", layout="wide")

    st.title("Quant Lab")
    st.caption(f"Version {__version__}")

    st.markdown(
        """
        Interactive quantitative finance experiments will live here.

        The first lab will cover Black-Scholes pricing, geometric Brownian
        motion, and Monte Carlo convergence.
        """
    )


if __name__ == "__main__":
    main()


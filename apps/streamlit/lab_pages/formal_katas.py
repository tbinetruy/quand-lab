from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("Formal Katas")
    st.caption("A deeper second pass through the mathematics behind the intro lab.")

    st.markdown(
        """
        The introductory track built a working map: options, probability,
        Brownian motion, GBM, Black-Scholes, Monte Carlo, PDEs, Greeks, and
        applied portfolio risk. The formal katas assume that map exists.

        The goal here is to rebuild the same machinery with more mathematical
        discipline. We will still use intuition, visuals, and implementation
        checks, but each kata should be a long-form course chapter rather than
        a short dashboard page.
        """
    )

    st.subheader("Course Contract")
    st.markdown(
        """
        Each formal kata should follow these rules:

        - start from a specific need created by the intro lab
        - define symbols before using them
        - avoid using concepts before they have been explained
        - mark unavoidable gaps as admissions or forward references
        - use visual arguments inside the text, not only charts at the end
        - prefer careful derivations over compressed formula lists
        - keep implementation as a verification tool, not the main exposition
        """
    )

    st.subheader("Page Pattern")
    st.markdown(
        """
        A typical kata should contain:

        1. Motivation From The Intro Lab
        2. What We Need To Explain
        3. Prerequisites From Earlier Katas
        4. Definitions and Notation
        5. Construction
        6. Visual Argument
        7. Propositions and Proof Sketches
        8. Worked Examples
        9. Implementation Check
        10. Common Pitfalls
        11. What We Are Still Admitting
        12. Where This Will Be Used Next
        """
    )

    st.subheader("Planned Katas")
    st.table(
        [
            {
                "Kata": "Analysis and Approximation",
                "Motivating intro need": "Taylor P&L, Greeks, finite differences.",
            },
            {
                "Kata": "Probability Foundations",
                "Motivating intro need": "Monte Carlo averages and expected payoff.",
            },
            {
                "Kata": "Stochastic Processes and Brownian Motion",
                "Motivating intro need": "GBM shocks and quadratic variation.",
            },
            {
                "Kata": "Stochastic Calculus",
                "Motivating intro need": "Ito's lemma in GBM and Black-Scholes.",
            },
            {
                "Kata": "Asset Modelling and GBM",
                "Motivating intro need": "The exponential simulator.",
            },
            {
                "Kata": "Pricing Measures and No-Arbitrage",
                "Motivating intro need": "Why pricing uses Q and r - q.",
            },
            {
                "Kata": "Black-Scholes Rebuilt",
                "Motivating intro need": "PDE, hedge, and closed-form formula.",
            },
            {
                "Kata": "Numerical Pricing Methods",
                "Motivating intro need": "Monte Carlo error and PDE grid error.",
            },
            {
                "Kata": "Portfolio Risk Rebuilt",
                "Motivating intro need": "Greek aggregation and scenario repricing.",
            },
        ]
    )

    st.subheader("Admissions")
    st.markdown(
        """
        We will still admit some results, especially where a full proof would
        require a separate course. The difference from the introductory track is
        that admissions must be named. If we use Brownian motion before proving
        its existence, or Girsanov's theorem before deriving it, the page should
        say exactly what is being admitted and why it is acceptable at that
        point in the course.
        """
    )

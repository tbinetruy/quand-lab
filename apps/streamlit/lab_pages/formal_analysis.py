from __future__ import annotations

from collections.abc import Callable
from math import exp

from components.charts import ChartRow, line_chart_spec, linspace, padded_range

import streamlit as st
import streamlit.components.v1 as st_components

Function = Callable[[float], float]


def render() -> None:
    st.header("Analysis and Approximation Kata")
    st.caption("Local linearity, Taylor expansion, curvature, and finite differences.")

    st.markdown(
        """
        This is the first formal kata. It assumes the introductory track has
        already shown why approximation matters: Greeks are derivatives, the
        Applied Risk page used a Taylor P&L estimate, and the PDE Solver
        replaced derivatives by grid differences. Here we rebuild those tools
        from the ground up.
        """
    )

    st.subheader("Motivation From The Intro Lab")
    st.markdown(
        """
        The introductory lab used analysis in three places:

        - **Greeks**: delta, gamma, vega, theta, and rho were derivatives of a
          price function.
        - **Applied Risk**: portfolio P&L was estimated by a Taylor expansion.
        - **PDE Solver**: derivatives were approximated by finite differences
          on a grid.

        Those are not separate tricks. They are the same idea repeated at
        different scales: understand a complicated function by looking near one
        point.
        """
    )

    st.subheader("What We Need To Explain")
    st.markdown(
        """
        We need answers to four questions.

        1. What does it mean for a derivative to be the best local linear
           approximation?
        2. Why does a second derivative measure curvature?
        3. Where does the Taylor approximation used for local P&L come from?
        4. Why do finite-difference formulas recover derivatives from nearby
           grid values?

        We will use one-dimensional notation first. Later katas will reuse the
        same ideas for option prices $V(t,S)$ and portfolio values
        $\\Pi(S,\\sigma,t,r)$.
        """
    )

    st.subheader("Prerequisites From Earlier Katas")
    st.markdown(
        """
        This is the first formal kata, so there are no formal prerequisites yet.
        From the intro lab, we assume only the following vocabulary:

        - an option price can be viewed as a function of inputs
        - a Greek is a sensitivity of that function
        - a numerical solver may replace a derivative by nearby values

        We will not use probability, Brownian motion, Ito calculus, or
        Black-Scholes in this kata except as motivation.
        """
    )

    st.subheader("Definitions and Notation")
    st.markdown(
        """
        Let $f$ be a real-valued function of one real variable:
        """
    )
    st.markdown(r"$$f:\mathbb{R}\to\mathbb{R}$$")
    st.markdown(
        """
        We choose a base point $x_0$ and study nearby inputs $x=x_0+h$.

        Symbols:

        - $x_0$: the point where we know the function locally
        - $h$: a small change in the input
        - $f'(x_0)$: first derivative at $x_0$
        - $f''(x_0)$: second derivative at $x_0$
        - $R_n(h)$: remainder after keeping terms up to order $n$

        In option language, $x$ might be spot $S$, and $f(x)$ might be an
        option price or portfolio value.
        """
    )

    st.subheader("Construction: Local Linearity")
    st.markdown(
        """
        A derivative is not just a slope drawn on a chart. It is a statement
        that, sufficiently close to $x_0$, the function behaves like a line:
        """
    )
    st.markdown(r"$$f(x_0+h)\approx f(x_0)+f'(x_0)h$$")
    st.markdown(
        """
        More formally, differentiability at $x_0$ means that the error in this
        linear approximation is small compared with $h$:
        """
    )
    st.markdown(
        "$$"
        r"f(x_0+h)=f(x_0)+f'(x_0)h+R_1(h),"
        r"\qquad"
        r"\frac{R_1(h)}{h}\to0\quad\text{as }h\to0"
        "$$"
    )
    st.markdown(
        """
        This notation is doing important work. It does not say the function is
        globally a line. It says that if we zoom in far enough around $x_0$,
        the remaining error becomes negligible relative to the step size.
        """
    )
    st_components.html(_tangent_diagram_markup(), height=360, scrolling=False)

    st.subheader("Visual Argument: Zooming Into a Curve")
    st.markdown(
        """
        The chart below uses a curved function and its tangent line at the base
        point. Near the base point, the tangent is useful. Far away, it can be
        misleading. This is the same limitation we saw when delta alone failed
        to track full repricing over larger spot moves.
        """
    )
    tangent_rows = _tangent_rows(base_x=1.0)
    tangent_min, tangent_max = padded_range([float(row["value"]) for row in tangent_rows])
    st.vega_lite_chart(
        tangent_rows,
        line_chart_spec(
            x_field="x",
            x_title="Input x",
            y_field="value",
            y_title="Value",
            y_min=tangent_min,
            y_max=tangent_max,
        ),
        use_container_width=True,
    )

    st.subheader("Proposition: Taylor's Formula to Second Order")
    st.markdown(
        """
        **Proposition.** If $f$ has two derivatives near $x_0$, then for small
        $h$:
        """
    )
    st.markdown(
        "$$"
        r"f(x_0+h)"
        r"="
        r"\underbrace{f(x_0)}_{\text{current value}}"
        r"+"
        r"\underbrace{f'(x_0)h}_{\text{linear change}}"
        r"+"
        r"\underbrace{\frac{1}{2}f''(x_0)h^2}_{\text{curvature correction}}"
        r"+"
        r"\underbrace{R_2(h)}_{\text{remaining error}}"
        "$$"
    )
    st.markdown(
        """
        If $f$ has a well-behaved third derivative, then the remaining error is
        of order $h^3$. We do not need the full theorem yet. For now, the key
        idea is that adding the second derivative improves a local line into a
        local parabola.
        """
    )

    st.subheader("Proof Sketch")
    st.markdown(
        """
        The number $h$ is the total input move. We start at $x_0$ and end at
        $x_0+h$. To walk along that interval, introduce a new variable $s$:
        """
    )
    st.markdown(r"$$x(s)=x_0+s,\qquad 0\le s\le h$$")
    st.markdown(
        """
        So $s=0$ means "we are at $x_0$," and $s=h$ means "we are at
        $x_0+h$." The variable $s$ is just a ruler measuring how far we have
        walked from the starting point.

        The first derivative gives the best local line for $f$. The same idea
        can be applied to the slope function $f'$. Near $x_0$, approximate the
        slope at the walking point $x_0+s$ by a local line:
        """
    )
    st.markdown(r"$$f'(x_0+s)\approx f'(x_0)+f''(x_0)s$$")
    st.markdown(
        """
        Split this walk into small pieces of width $ds$. Over one small piece,
        the input moves from $x_0+s$ to $x_0+s+ds$. The derivative says:
        """
    )
    st.markdown(r"$$f(x_0+s+ds)-f(x_0+s)\approx f'(x_0+s)\,ds$$")
    st.markdown(
        """
        Now add these tiny changes along the whole interval. The left-hand side
        telescopes: every intermediate value cancels, leaving only final value
        minus initial value:
        """
    )
    st.markdown(
        "$$"
        r"\sum \left(f(x_0+s+ds)-f(x_0+s)\right)"
        r"\longrightarrow f(x_0+h)-f(x_0)"
        "$$"
    )
    st.markdown(
        """
        The right-hand side becomes the integral of the slope:
        """
    )
    st.markdown(r"$$f(x_0+h)-f(x_0)=\int_0^h f'(x_0+s)\,ds$$")
    st.markdown(
        """
        Substitute the local approximation for the slope:
        """
    )
    st.markdown(
        "$$"
        r"\int_0^h \left(f'(x_0)+f''(x_0)s\right)\,ds"
        r"="
        r"f'(x_0)h+\frac{1}{2}f''(x_0)h^2"
        "$$"
    )
    st.markdown(
        """
        This is why the second-order term has the factor $1/2$: it comes from
        integrating a linearly changing slope.
        """
    )

    st.subheader("Worked Example: Local Portfolio P&L")
    st.markdown(
        """
        In Applied Risk, the portfolio value was written as $\\Pi(S)$ when we
        focused only on spot. Taylor's formula gives:
        """
    )
    st.markdown(
        "$$"
        r"\Pi(S+\Delta S)-\Pi(S)"
        r"\approx"
        r"\Delta_\Pi\Delta S"
        r"+\frac{1}{2}\Gamma_\Pi(\Delta S)^2"
        "$$"
    )
    st.markdown(
        """
        This is just the same formula with names changed:
        """
    )
    st.table(
        [
            {"Analysis symbol": "$f$", "Portfolio symbol": "$\\Pi$", "Meaning": "value function"},
            {"Analysis symbol": "$x$", "Portfolio symbol": "$S$", "Meaning": "spot input"},
            {"Analysis symbol": "$h$", "Portfolio symbol": "$\\Delta S$", "Meaning": "spot move"},
            {
                "Analysis symbol": "$f'(x)$",
                "Portfolio symbol": "$\\Delta_\\Pi$",
                "Meaning": "delta",
            },
            {
                "Analysis symbol": "$f''(x)$",
                "Portfolio symbol": "$\\Gamma_\\Pi$",
                "Meaning": "gamma",
            },
        ]
    )

    st.subheader("Visual Argument: Why Gamma Is Curvature")
    st.markdown(
        """
        A positive second derivative means the slope increases as $x$ increases.
        The graph bends upward. A negative second derivative means the slope
        decreases and the graph bends downward. For options, gamma measures this
        curvature with respect to spot.
        """
    )
    st_components.html(_curvature_diagram_markup(), height=360, scrolling=False)

    st.subheader("Finite Differences")
    st.markdown(
        """
        Derivatives are limits. Computers do not take limits directly; they
        evaluate functions at nearby points. Let a grid have spacing $h$:
        """
    )
    st.markdown(r"$$x_{i-1}=x_i-h,\qquad x_{i+1}=x_i+h$$")
    st.markdown(
        """
        A right-sided slope is:
        """
    )
    st.markdown(r"$$\frac{f(x_i+h)-f(x_i)}{h}$$")
    st.markdown(
        """
        A left-sided slope is:
        """
    )
    st.markdown(r"$$\frac{f(x_i)-f(x_i-h)}{h}$$")
    st.markdown(
        """
        A symmetric first derivative uses both sides:
        """
    )
    st.markdown(r"$$f'(x_i)\approx\frac{f(x_i+h)-f(x_i-h)}{2h}$$")
    st.markdown(
        """
        The second derivative is change in slope divided by distance. Subtract
        the left slope from the right slope, then divide by $h$:
        """
    )
    st.markdown(
        "$$"
        r"f''(x_i)\approx"
        r"\frac{\frac{f(x_i+h)-f(x_i)}{h}-\frac{f(x_i)-f(x_i-h)}{h}}{h}"
        r"="
        r"\frac{f(x_i+h)-2f(x_i)+f(x_i-h)}{h^2}"
        "$$"
    )
    st_components.html(_stencil_diagram_markup(), height=330, scrolling=False)

    st.subheader("Implementation Check")
    st.markdown(
        """
        We can test these formulas on a function whose derivatives we know:
        """
    )
    st.markdown(r"$$f(x)=e^x,\qquad f'(x)=e^x,\qquad f''(x)=e^x$$")
    st.markdown(
        """
        The table evaluates finite differences at $x=0$. As the grid spacing
        gets smaller, the numerical estimates approach the true value $1$.
        """
    )
    st.table(_finite_difference_error_rows())

    st.subheader("Common Pitfalls")
    st.markdown(
        """
        - **Local does not mean global.** A tangent line can be excellent near
          $x_0$ and useless far away.
        - **Small input moves are not dimensionless.** A small move in spot may
          be large relative to volatility, maturity, or moneyness.
        - **Second order is not exact.** Gamma improves the approximation, but
          higher-order terms still exist.
        - **Finite differences see the grid, not the continuum.** A coarse grid
          may miss sharp curvature, especially near payoff kinks.
        """
    )

    st.subheader("What We Are Still Admitting")
    st.markdown(
        """
        We are not proving the full Taylor theorem with all regularity
        conditions. We are using the version needed for the rest of the course:
        when a function is sufficiently smooth near a point, its local behavior
        can be approximated by a polynomial whose coefficients are derivatives.

        We are also not yet discussing multivariable error terms in detail.
        Applied Risk used several inputs at once, but the one-dimensional case
        is the clean foundation. We will return to multivariable approximations
        when rebuilding portfolio risk.
        """
    )

    st.subheader("Where This Will Be Used Next")
    st.markdown(
        """
        This kata will be reused immediately:

        - Probability will use approximation when discussing sample averages.
        - Brownian motion will force us to revisit why ordinary Taylor rules
          fail for rough paths.
        - Stochastic calculus will keep one second-order term because Brownian
          increments have nonzero quadratic variation.
        - Black-Scholes will apply Ito's lemma to $V(t,S_t)$.
        - Numerical methods will reuse the finite-difference stencils.
        """
    )


def _model_function(x: float) -> float:
    return 0.08 * (x - 1.0) ** 3 + 0.7 * (x - 1.0) ** 2 + 1.2 * (x - 1.0) + 3.0


def _model_derivative(base_x: float) -> float:
    return 0.24 * (base_x - 1.0) ** 2 + 1.4 * (base_x - 1.0) + 1.2


def _tangent_value(x: float, *, base_x: float) -> float:
    return _model_function(base_x) + _model_derivative(base_x) * (x - base_x)


def _tangent_rows(*, base_x: float) -> list[ChartRow]:
    rows: list[ChartRow] = []
    for x in linspace(-2.0, 4.0, 160):
        rows.append({"x": x, "value": _model_function(x), "series": "Function"})
        rows.append({"x": x, "value": _tangent_value(x, base_x=base_x), "series": "Tangent"})
    return rows


def _finite_difference_error_rows() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    true_value = 1.0
    x = 0.0

    for h in [0.5, 0.25, 0.1, 0.05, 0.01]:
        central_first = (exp(x + h) - exp(x - h)) / (2.0 * h)
        central_second = (exp(x + h) - 2.0 * exp(x) + exp(x - h)) / h**2
        rows.append(
            {
                "h": h,
                "first derivative estimate": central_first,
                "first derivative error": abs(central_first - true_value),
                "second derivative estimate": central_second,
                "second derivative error": abs(central_second - true_value),
            }
        )

    return rows


def _tangent_diagram_markup() -> str:
    return """
    <svg viewBox="0 0 760 320" width="100%" height="320" role="img"
         aria-label="A curve with a tangent line at a base point.">
      <rect x="20" y="20" width="720" height="280" fill="#f8fafc" stroke="#e2e8f0" rx="8" />
      <line x1="80" y1="250" x2="700" y2="250" stroke="#334155" stroke-width="2" />
      <line x1="100" y1="270" x2="100" y2="55" stroke="#334155" stroke-width="2" />
      <path d="M105 232 C205 232, 275 196, 350 145 C435 88, 540 72, 680 66"
            fill="none" stroke="#2563eb" stroke-width="4" />
      <line x1="230" y1="211" x2="560" y2="31" stroke="#f97316" stroke-width="3"
            stroke-dasharray="8 8" />
      <circle cx="350" cy="145" r="8" fill="#dc2626" />
      <line x1="350" y1="145" x2="350" y2="250" stroke="#94a3b8" stroke-width="2"
            stroke-dasharray="5 7" />
      <text x="365" y="175" font-size="16" fill="#334155">base point x0</text>
      <text x="585" y="70" font-size="16" fill="#2563eb">f(x)</text>
      <text x="510" y="48" font-size="16" fill="#f97316">tangent line</text>
      <text x="372" y="244" font-size="14" fill="#64748b">zoom here: curve ≈ line</text>
      <text x="90" y="292" font-size="14" fill="#334155">input x</text>
      <text x="28" y="70" font-size="14" fill="#334155">value</text>
    </svg>
    """


def _curvature_diagram_markup() -> str:
    return """
    <svg viewBox="0 0 760 330" width="100%" height="330" role="img"
         aria-label="Convex and concave curvature diagrams.">
      <rect x="20" y="20" width="330" height="285" fill="#f8fafc" stroke="#e2e8f0" rx="8" />
      <rect x="410" y="20" width="330" height="285" fill="#f8fafc" stroke="#e2e8f0" rx="8" />
      <text x="95" y="54" font-size="17" fill="#334155">positive curvature</text>
      <text x="96" y="82" font-size="14" fill="#334155">tangent slope increases left to right</text>
      <path d="M75 250 C145 246, 225 195, 320 86" fill="none" stroke="#16a34a" stroke-width="4" />
      <line x1="82" y1="251" x2="140" y2="242" stroke="#334155" stroke-width="2"
            stroke-dasharray="7 7" />
      <line x1="172" y1="214" x2="240" y2="170" stroke="#334155" stroke-width="2"
            stroke-dasharray="7 7" />
      <line x1="258" y1="140" x2="322" y2="68" stroke="#334155" stroke-width="2"
            stroke-dasharray="7 7" />
      <text x="122" y="286" font-size="15" fill="#16a34a">f''(x) &gt; 0</text>

      <text x="485" y="54" font-size="17" fill="#334155">negative curvature</text>
      <text x="488" y="82" font-size="14" fill="#334155">
        tangent slope decreases left to right
      </text>
      <path d="M455 250 C500 135, 590 95, 705 86" fill="none" stroke="#9333ea" stroke-width="4" />
      <line x1="448" y1="268" x2="495" y2="153" stroke="#334155" stroke-width="2"
            stroke-dasharray="7 7" />
      <line x1="535" y1="112" x2="610" y2="92" stroke="#334155" stroke-width="2"
            stroke-dasharray="7 7" />
      <line x1="635" y1="90" x2="705" y2="84" stroke="#334155" stroke-width="2"
            stroke-dasharray="7 7" />
      <text x="512" y="286" font-size="15" fill="#9333ea">f''(x) &lt; 0</text>
    </svg>
    """


def _stencil_diagram_markup() -> str:
    return """
    <svg viewBox="0 0 760 300" width="100%" height="300" role="img"
         aria-label="Finite difference stencil using left, center, and right grid points.">
      <rect x="35" y="30" width="690" height="230" fill="#f8fafc" stroke="#e2e8f0" rx="8" />
      <line x1="105" y1="170" x2="655" y2="170" stroke="#334155" stroke-width="3" />
      <circle cx="200" cy="170" r="11" fill="#64748b" />
      <circle cx="380" cy="170" r="13" fill="#2563eb" />
      <circle cx="560" cy="170" r="11" fill="#64748b" />
      <line x1="200" y1="170" x2="380" y2="170" stroke="#f97316" stroke-width="5"
            stroke-dasharray="8 8" />
      <line x1="380" y1="170" x2="560" y2="170" stroke="#16a34a" stroke-width="5"
            stroke-dasharray="8 8" />
      <text x="176" y="210" font-size="17" fill="#334155">xᵢ₋₁</text>
      <text x="365" y="210" font-size="17" fill="#2563eb">xᵢ</text>
      <text x="537" y="210" font-size="17" fill="#334155">xᵢ₊₁</text>
      <text x="278" y="150" font-size="15" fill="#f97316">left slope</text>
      <text x="430" y="150" font-size="15" fill="#16a34a">right slope</text>
      <path d="M200 112 L380 78 L560 118" fill="none" stroke="#0f172a" stroke-width="3" />
      <circle cx="200" cy="112" r="6" fill="#0f172a" />
      <circle cx="380" cy="78" r="6" fill="#0f172a" />
      <circle cx="560" cy="118" r="6" fill="#0f172a" />
      <text x="190" y="92" font-size="15" fill="#0f172a">fᵢ₋₁</text>
      <text x="370" y="58" font-size="15" fill="#0f172a">fᵢ</text>
      <text x="550" y="98" font-size="15" fill="#0f172a">fᵢ₊₁</text>
      <text x="118" y="248" font-size="15" fill="#334155">
        second difference = right slope - left slope, divided by grid spacing
      </text>
    </svg>
    """

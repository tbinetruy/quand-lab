from __future__ import annotations

from components.charts import (
    ChartRow,
    line_chart_spec,
    padded_range,
    surface_chart_spec,
    surface_value_max,
    surface_value_min,
)
from components.source import show_source_file

import streamlit as st
import streamlit.components.v1 as st_components
from quant_lab.domain import BlackScholesMarket, EuropeanOption, OptionType
from quant_lab.pricing import (
    FiniteDifferenceConfig,
    FiniteDifferenceMethod,
    FiniteDifferenceResult,
    black_scholes_price,
    finite_difference_price,
)


def render() -> None:
    st.header("Finite-Difference PDE Solver")
    st.caption("Price European options by discretizing the Black-Scholes PDE.")

    st.markdown(
        """
        Instead of simulating paths, finite-difference solvers approximate the
        Black-Scholes PDE on a grid of stock prices and time-to-expiry values.
        The result is a full option-value surface, not just one price.

        We care about PDE solvers even after Monte Carlo because they answer a
        different kind of question. Monte Carlo is very flexible and scales well
        when there are many sources of randomness, but it gives noisy estimates
        at selected points. A PDE solver gives a whole structured surface over
        state and time. That makes local behavior, boundaries, early-exercise
        extensions, and numerical error much easier to inspect.
        """
    )

    st.subheader("Concept")
    st.markdown(
        """
        The Black-Scholes derivation gave a PDE in calendar time $t$:
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
        For numerical work it is often clearer to use time-to-expiry:
        """
    )
    st.markdown(r"$$\tau=T-t$$")
    st.markdown(
        """
        The reason is practical. In calendar time, the payoff is known at the
        final time:
        """
    )
    st.markdown(r"$$V(T,S)=\Phi(S)$$")
    st.markdown(
        """
        That is a backward problem: we know the end and solve back to today.
        With time-to-expiry, expiry becomes $\\tau=0$, so the payoff becomes the
        first row of the numerical grid:
        """
    )
    st.markdown(r"$$V(S,0)=\Phi(S)$$")
    st.markdown(
        """
        The solver can then march forward in $\\tau$ from the known payoff
        toward today's maturity. This also gives the PDE the standard evolution
        form "next time row from current time row."
        """
    )
    st.markdown(
        """
        Expiry is then $\\tau=0$, and today's value for a maturity-$T$ option is
        found at $\\tau=T$. Since $\\partial V/\\partial t=-\\partial V/\\partial
        \\tau$, the PDE becomes:
        """
    )
    st.markdown(
        r"$$\frac{\partial V}{\partial \tau}="
        r"\frac{1}{2}\sigma^2 S^2\frac{\partial^2 V}{\partial S^2}"
        r"+(r-q)S\frac{\partial V}{\partial S}-rV$$"
    )
    st.markdown(
        """
        This is the equation the solver marches forward in $\\tau$: start from
        the payoff at expiry, then evolve the value surface toward larger
        time-to-expiry.
        """
    )

    st.subheader("Grid")
    st.markdown(
        """
        The continuous variables are replaced by grid points:
        """
    )
    st.markdown(r"$$S_i=i\Delta S,\qquad \tau_n=n\Delta\tau$$")
    st.markdown(
        """
        The numerical value at grid point $(S_i,\tau_n)$ is written:
        """
    )
    st.markdown(r"$$V_i^n\approx V(S_i,\tau_n)$$")
    st.markdown(
        """
        This notation means:

        - $i$ moves across stock-price grid points
        - $n$ moves across time-to-expiry grid points
        - $V_i^0$ is known from the payoff
        - $V_i^N$ is the option value curve for today's maturity
        """
    )
    st_components.html(_grid_diagram_markup(), height=430, scrolling=False)

    st.subheader("Initial and Boundary Conditions")
    st.markdown(
        """
        In time-to-expiry, the "initial" condition is the terminal payoff:
        """
    )
    st.markdown(r"$$V(S,0)=\Phi(S)$$")
    st.markdown(
        """
        For a call and put:
        """
    )
    st.markdown(r"$$\Phi_{call}(S)=\max(S-K,0),\qquad \Phi_{put}(S)=\max(K-S,0)$$")
    st.markdown(
        """
        The stock grid is finite, so we also need boundary conditions at
        $S=0$ and at a large artificial upper boundary $S_{max}$.

        Near zero spot:

        - a call is nearly worthless
        - a put behaves like a discounted strike

        Near very large spot:

        - a call behaves like stock minus discounted strike
        - a put is nearly worthless
        """
    )
    st.markdown(r"$$C(0,\tau)=0,\qquad C(S_{max},\tau)\approx S_{max}e^{-q\tau}-Ke^{-r\tau}$$")
    st.markdown(r"$$P(0,\tau)\approx Ke^{-r\tau},\qquad P(S_{max},\tau)=0$$")

    st.subheader("Finite Differences")
    st.markdown(
        """
        Derivatives are replaced by local differences. At an interior grid point:
        """
    )
    st.markdown(
        """
        For the first derivative, a one-sided slope would use only the point to
        the right:
        """
    )
    st.markdown(r"$$\frac{\partial V}{\partial S}\approx\frac{V_{i+1}^n-V_i^n}{\Delta S}$$")
    st.markdown(
        """
        Inside the grid we can do better by using a symmetric interval around
        $S_i$. The distance from $S_{i-1}$ to $S_{i+1}$ is $2\\Delta S$, so:
        """
    )
    st.markdown(r"$$\frac{\partial V}{\partial S}\approx\frac{V_{i+1}^n-V_{i-1}^n}{2\Delta S}$$")
    st.markdown(
        """
        The second derivative measures how quickly the slope changes. Approximate
        the slope on the right and the slope on the left:
        """
    )
    st.markdown(r"$$\text{right slope}\approx\frac{V_{i+1}^n-V_i^n}{\Delta S}$$")
    st.markdown(r"$$\text{left slope}\approx\frac{V_i^n-V_{i-1}^n}{\Delta S}$$")
    st.markdown(
        """
        Then divide their difference by $\\Delta S$:
        """
    )
    st.markdown(
        r"$$\frac{\partial^2V}{\partial S^2}\approx"
        r"\frac{\frac{V_{i+1}^n-V_i^n}{\Delta S}"
        r"-\frac{V_i^n-V_{i-1}^n}{\Delta S}}{\Delta S}$$"
    )
    st.markdown(
        """
        Simplifying gives the standard central second-difference stencil:
        """
    )
    st.markdown(
        r"$$\frac{\partial^2V}{\partial S^2}\approx"
        r"\frac{V_{i+1}^n-2V_i^n+V_{i-1}^n}{\Delta S^2}$$"
    )
    st.markdown(
        """
        In our implementation the grid is uniform in $S$, so those finite
        differences become a tridiagonal stencil: each interior value talks only
        to its left neighbor, itself, and its right neighbor.
        """
    )

    st.subheader("Schemes")
    st.markdown(
        """
        The schemes differ in where they evaluate the spatial operator. Let
        $L(V)$ denote the Black-Scholes spatial operator:
        """
    )
    st.markdown(
        "$$"
        r"L(V)=\frac{1}{2}\sigma^2S^2V_{SS}+(r-q)SV_S-rV"
        "$$"
    )
    st.markdown(
        """
        The PDE is:
        """
    )
    st.markdown(r"$$V_\tau=L(V)$$")
    st.table(
        [
            {
                "Scheme": "Explicit",
                "Step idea": "$V^{n+1}=V^n+\\Delta\\tau L(V^n)$",
                "Tradeoff": "Simple and transparent, but stability-limited.",
            },
            {
                "Scheme": "Implicit",
                "Step idea": "$V^{n+1}=V^n+\\Delta\\tau L(V^{n+1})$",
                "Tradeoff": "Requires solving a linear system, but is much more stable.",
            },
            {
                "Scheme": "Crank-Nicolson",
                "Step idea": "$V^{n+1}=V^n+\\frac{\\Delta\\tau}{2}(L(V^n)+L(V^{n+1}))$",
                "Tradeoff": "Averages explicit and implicit steps; usually more accurate.",
            },
        ]
    )
    st.markdown(
        """
        Explicit stepping computes the next row directly from the current row.
        Implicit and Crank-Nicolson methods solve a tridiagonal linear system at
        each time step. That extra work buys stability and accuracy.
        """
    )

    st.subheader("Error Sources")
    st.markdown(
        """
        Numerical error does not come from one place:

        - spot discretization: $\\Delta S$ controls how well we resolve curvature
        - time discretization: $\\Delta\\tau$ controls stepping error
        - payoff kink: calls and puts are not smooth at $S=K$ at expiry
        - boundary truncation: $S_{max}$ is finite even though the model allows
          arbitrarily large prices
        - scheme behavior: explicit can become unstable; Crank-Nicolson can show
          small oscillations near nonsmooth initial data

        This is why the error curve often has its largest structure near the
        strike: the payoff kink is exactly where the finite-difference stencil
        has the hardest local approximation problem.
        """
    )

    st.subheader("Monte Carlo vs PDE")
    st.markdown(
        """
        Neither method dominates everywhere.

        PDE methods are attractive when:

        - the number of state variables is small
        - we want the whole value surface
        - boundary behavior matters
        - we need smooth local structure for Greeks
        - we later want early-exercise or free-boundary ideas, as with
          American-style options

        Monte Carlo is attractive when:

        - the payoff is path-dependent, meaning it depends on the history of the
          path, not only $S_T$
        - examples include Asian options, lookback options, and barrier options
        - there are many risk factors
        - the state dimension would make a PDE grid explode
        - a noisy estimate is acceptable or can be reduced with variance
          reduction

        Early exercise and path dependence are different ideas. American
        options are early-exercise products. Asian, lookback, and barrier
        options are path-dependent products. Some contracts can have both.
        """
    )

    st.subheader("Implementation")
    show_source_file("src/quant_lab/pricing/finite_difference.py")

    st.subheader("Experiment")
    left, right = st.columns(2)

    with left:
        selected_option_type = st.segmented_control(
            "Option type",
            options=["call", "put"],
            format_func=lambda value: value.title(),
            default="call",
            key="pde_option_type",
        )
        option_type = OptionType(selected_option_type or "call")
        selected_method = st.selectbox(
            "Scheme",
            options=list(FiniteDifferenceMethod),
            format_func=lambda value: value.value.replace("_", " ").title(),
            index=2,
        )
        strike = st.number_input("Strike", min_value=0.01, value=100.0, step=1.0)
        maturity = st.number_input("Maturity in years", min_value=0.0, value=1.0, step=0.25)

    with right:
        spot = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0)
        rate = st.number_input("Risk-free rate", value=0.05, step=0.01, format="%.4f")
        dividend_yield = st.number_input("Dividend yield", value=0.0, step=0.01, format="%.4f")
        volatility = st.number_input("Volatility", min_value=0.0, value=0.2, step=0.05)
        n_spot_steps = st.number_input(
            "Spot steps",
            min_value=20,
            max_value=300,
            value=100,
            step=10,
        )
        n_time_steps = st.number_input(
            "Time steps",
            min_value=10,
            max_value=5_000,
            value=300,
            step=50,
        )

    option = EuropeanOption(option_type=option_type, strike=strike, maturity_years=maturity)
    market = BlackScholesMarket(
        spot=spot,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    config = FiniteDifferenceConfig(
        n_spot_steps=n_spot_steps,
        n_time_steps=n_time_steps,
        method=FiniteDifferenceMethod(selected_method or FiniteDifferenceMethod.CRANK_NICOLSON),
    )
    result = finite_difference_price(option, market, config)
    analytical_price = black_scholes_price(option, market)

    metric_left, metric_middle, metric_right = st.columns(3)
    metric_left.metric("PDE price", f"{result.price:.4f}")
    metric_middle.metric("Black-Scholes price", f"{analytical_price:.4f}")
    metric_right.metric("Absolute error", f"{abs(result.price - analytical_price):.4f}")

    st.json(
        {
            "scheme": result.method.value,
            "stability_ratio": round(result.stability_ratio, 6),
            "explicit_scheme_note": _stability_note(result),
        }
    )

    st.markdown(
        """
        Option value surface:

        The raw option value is dominated by intrinsic value as spot increases,
        so time effects can be visually subtle on the full stock grid. Read
        this as the numerical solution $V(S,\\tau)$ over the full grid.
        """
    )
    value_surface_rows = _value_surface_rows(option, result)
    st.vega_lite_chart(
        value_surface_rows,
        surface_chart_spec(
            value_title="Option value",
            value_min=surface_value_min(value_surface_rows, field="value"),
            value_max=surface_value_max(value_surface_rows, field="value"),
            y_field="time_to_expiry",
            y_title="Time to expiry",
            value_field="value",
        ),
        use_container_width=True,
    )

    st.markdown(
        """
        Time value surface:

        This subtracts intrinsic value from the option value. It makes the
        optionality component visible: time value is usually concentrated near
        the strike because that is where future moves can most change the payoff
        outcome.
        """
    )
    st.vega_lite_chart(
        value_surface_rows,
        surface_chart_spec(
            value_title="Option value - intrinsic value",
            value_min=surface_value_min(value_surface_rows, field="time_value"),
            value_max=surface_value_max(value_surface_rows, field="time_value"),
            y_field="time_to_expiry",
            y_title="Time to expiry",
            value_field="time_value",
        ),
        use_container_width=True,
    )

    st.markdown(
        """
        Final price curve:

        This compares the last PDE time row, at the selected maturity, with the
        Black-Scholes closed form. If the grid is fine enough and the scheme is
        stable, the two curves should nearly overlap.
        """
    )
    final_curve_rows = _final_curve_rows(option, market, result)
    final_values = [float(row["value"]) for row in final_curve_rows]
    y_min, y_max = padded_range(final_values)
    st.vega_lite_chart(
        final_curve_rows,
        line_chart_spec(
            x_field="spot",
            x_title="Spot",
            y_field="value",
            y_title="Option value",
            y_min=y_min,
            y_max=y_max,
        ),
        use_container_width=True,
    )

    st.markdown(
        """
        Error against Black-Scholes:

        This isolates numerical error. Expect the most interesting structure
        near the strike, where the terminal payoff has a kink and the grid has
        to smooth that nonsmooth initial condition as it steps forward in
        time-to-expiry.
        """
    )
    error_rows = _error_rows(option, market, result)
    error_values = [float(row["error"]) for row in error_rows]
    error_min, error_max = padded_range([*error_values, 0.0])
    st.vega_lite_chart(
        error_rows,
        line_chart_spec(
            x_field="spot",
            x_title="Spot",
            y_field="error",
            y_title="PDE - Black-Scholes",
            y_min=error_min,
            y_max=error_max,
            color_field=None,
        ),
        use_container_width=True,
    )


def _stability_note(result: FiniteDifferenceResult) -> str:
    if result.method is not FiniteDifferenceMethod.EXPLICIT:
        return (
            "Implicit and Crank-Nicolson schemes are not constrained by the same "
            "explicit stability limit."
        )
    if result.stability_ratio <= 1.0:
        return "Explicit scheme is within the rough sigma^2 * M^2 * dt <= 1 stability guideline."
    return "Explicit scheme exceeds the rough sigma^2 * M^2 * dt <= 1 stability guideline."


def _grid_diagram_markup() -> str:
    return """
    <div style="width:100%; overflow-x:auto; margin: 0.75rem 0 1rem 0;">
      <svg viewBox="0 0 820 400" role="img" aria-label="Finite-difference grid diagram"
           style="width:100%; min-width:640px; max-width:900px; height:auto;
                  border:1px solid #d9dee7; border-radius:8px; background:#ffffff;">
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3"
                  orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#334155" />
          </marker>
        </defs>

        <rect x="72" y="52" width="560" height="220" fill="#f8fafc" stroke="#e2e8f0" />

        <line x1="72" y1="272" x2="632" y2="272" stroke="#334155" stroke-width="2"
              marker-end="url(#arrow)" />
        <line x1="72" y1="272" x2="72" y2="52" stroke="#334155" stroke-width="2"
              marker-end="url(#arrow)" />

        <text x="346" y="342" text-anchor="middle" fill="#334155" font-size="15">
          stock grid: Sᵢ = i ΔS
        </text>
        <text x="24" y="166" text-anchor="middle" fill="#334155" font-size="15"
              transform="rotate(-90 24 166)">
          time-to-expiry: τₙ = n Δτ
        </text>

        <line x1="72" y1="272" x2="632" y2="272" stroke="#cbd5e1" stroke-width="3" />
        <text x="654" y="277" fill="#475569" font-size="14">payoff row: Vᵢ⁰ = Φ(Sᵢ)</text>

        <line x1="72" y1="52" x2="632" y2="52" stroke="#93c5fd" stroke-width="3" />
        <text x="654" y="57" fill="#2563eb" font-size="14">today row: Vᵢᴺ</text>

        <g stroke="#cbd5e1" stroke-width="1">
          <line x1="72" y1="217" x2="632" y2="217" />
          <line x1="72" y1="162" x2="632" y2="162" />
          <line x1="72" y1="107" x2="632" y2="107" />
          <line x1="184" y1="52" x2="184" y2="272" />
          <line x1="296" y1="52" x2="296" y2="272" />
          <line x1="408" y1="52" x2="408" y2="272" />
          <line x1="520" y1="52" x2="520" y2="272" />
          <line x1="632" y1="52" x2="632" y2="272" />
        </g>

        <g fill="#64748b">
          <circle cx="72" cy="272" r="4" />
          <circle cx="184" cy="272" r="4" />
          <circle cx="296" cy="272" r="4" />
          <circle cx="408" cy="272" r="4" />
          <circle cx="520" cy="272" r="4" />
          <circle cx="632" cy="272" r="4" />
          <circle cx="72" cy="217" r="4" />
          <circle cx="184" cy="217" r="4" />
          <circle cx="296" cy="217" r="4" />
          <circle cx="408" cy="217" r="4" />
          <circle cx="520" cy="217" r="4" />
          <circle cx="632" cy="217" r="4" />
          <circle cx="72" cy="162" r="4" />
          <circle cx="184" cy="162" r="4" />
          <circle cx="296" cy="162" r="4" />
          <circle cx="408" cy="162" r="4" />
          <circle cx="520" cy="162" r="4" />
          <circle cx="632" cy="162" r="4" />
          <circle cx="72" cy="107" r="4" />
          <circle cx="184" cy="107" r="4" />
          <circle cx="296" cy="107" r="4" />
          <circle cx="408" cy="107" r="4" />
          <circle cx="520" cy="107" r="4" />
          <circle cx="632" cy="107" r="4" />
          <circle cx="72" cy="52" r="4" />
          <circle cx="184" cy="52" r="4" />
          <circle cx="296" cy="52" r="4" />
          <circle cx="408" cy="52" r="4" />
          <circle cx="520" cy="52" r="4" />
          <circle cx="632" cy="52" r="4" />
        </g>

        <g>
          <circle cx="408" cy="162" r="8" fill="#2563eb" />
          <circle cx="296" cy="162" r="7" fill="#f97316" />
          <circle cx="520" cy="162" r="7" fill="#f97316" />
          <circle cx="408" cy="107" r="7" fill="#16a34a" />
          <line x1="296" y1="162" x2="520" y2="162" stroke="#f97316" stroke-width="2"
                stroke-dasharray="5 4" />
          <line x1="408" y1="162" x2="408" y2="107" stroke="#16a34a" stroke-width="2"
                stroke-dasharray="5 4" />
          <text x="408" y="145" text-anchor="middle" fill="#1d4ed8" font-size="15">Vᵢⁿ</text>
          <text x="408" y="92" text-anchor="middle" fill="#15803d" font-size="13">next row</text>
          <text x="408" y="189" text-anchor="middle" fill="#c2410c" font-size="13">
            spatial stencil
          </text>
        </g>

        <text x="78" y="294" fill="#475569" font-size="13">n = 0</text>
        <text x="78" y="44" fill="#475569" font-size="13">n = N</text>
        <text x="72" y="318" text-anchor="middle" fill="#475569" font-size="13">i = 0</text>
        <text x="632" y="318" text-anchor="middle" fill="#475569" font-size="13">i = M</text>
      </svg>
    </div>
    """


def _value_surface_rows(
    option: EuropeanOption,
    result: FiniteDifferenceResult,
) -> list[ChartRow]:
    rows: list[ChartRow] = []
    spot_stride = max(1, len(result.spot_grid) // 60)
    time_stride = max(1, len(result.time_to_expiry_grid) // 60)

    for time_index in range(0, len(result.time_to_expiry_grid), time_stride):
        for spot_index in range(0, len(result.spot_grid), spot_stride):
            spot = float(result.spot_grid[spot_index])
            value = float(result.values[time_index, spot_index])
            intrinsic_value = _intrinsic_value(option, spot)
            rows.append(
                {
                    "spot": round(spot, 2),
                    "time_to_expiry": round(float(result.time_to_expiry_grid[time_index]), 4),
                    "value": round(value, 6),
                    "time_value": round(value - intrinsic_value, 6),
                    "option_type": option.option_type.value.title(),
                }
            )

    return rows


def _final_curve_rows(
    option: EuropeanOption,
    market: BlackScholesMarket,
    result: FiniteDifferenceResult,
) -> list[ChartRow]:
    rows: list[ChartRow] = []
    for spot, pde_value in zip(result.spot_grid, result.values[-1], strict=True):
        safe_spot = max(float(spot), 1e-12)
        analytical_market = _replace_market(market, spot=safe_spot)
        rows.append(
            {
                "spot": round(float(spot), 4),
                "series": "PDE",
                "value": float(pde_value),
                "option_type": option.option_type.value.title(),
            }
        )
        rows.append(
            {
                "spot": round(float(spot), 4),
                "series": "Black-Scholes",
                "value": black_scholes_price(option, analytical_market),
                "option_type": option.option_type.value.title(),
            }
        )

    return rows


def _error_rows(
    option: EuropeanOption,
    market: BlackScholesMarket,
    result: FiniteDifferenceResult,
) -> list[ChartRow]:
    rows: list[ChartRow] = []
    for spot, pde_value in zip(result.spot_grid, result.values[-1], strict=True):
        safe_spot = max(float(spot), 1e-12)
        analytical_market = _replace_market(market, spot=safe_spot)
        analytical_value = black_scholes_price(option, analytical_market)
        rows.append(
            {
                "spot": round(float(spot), 4),
                "error": float(pde_value) - analytical_value,
                "option_type": option.option_type.value.title(),
            }
        )

    return rows


def _intrinsic_value(option: EuropeanOption, spot: float) -> float:
    if option.option_type is OptionType.CALL:
        return max(spot - option.strike, 0.0)
    return max(option.strike - spot, 0.0)


def _replace_market(market: BlackScholesMarket, *, spot: float) -> BlackScholesMarket:
    return BlackScholesMarket(
        spot=spot,
        rate=market.rate,
        volatility=market.volatility,
        dividend_yield=market.dividend_yield,
    )

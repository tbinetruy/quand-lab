from __future__ import annotations

import numpy as np
from components.charts import ChartRow, ChartSpec, line_chart_spec, padded_range

import streamlit as st
from quant_lab.stochastic.brownian import brownian_increments


def render() -> None:
    st.header("Stochastic Calculus")
    st.caption("The calculus rules behind GBM and Black-Scholes derivations.")

    st.markdown(
        """
        Ordinary calculus is built for smooth paths. Brownian paths are
        continuous but extremely rough: they have no classical derivative and
        their small increments accumulate differently from smooth-function
        increments.

        This page focuses on the machinery itself: stochastic integrals,
        quadratic variation, Ito processes, and Ito's lemma. We will apply these
        tools in the dedicated GBM and Black-Scholes pages.
        """
    )

    st.subheader("Why Ordinary Calculus Breaks")
    st.markdown(
        """
        For a smooth path, the sum of squared increments goes to zero as the
        time grid gets finer. For Brownian motion, the sum of squared increments
        converges to elapsed time:
        """
    )
    st.markdown(r"$$\sum_i (W_{t_{i+1}}-W_{t_i})^2 \rightarrow T$$")
    st.markdown(
        """
        This is quadratic variation. It is the reason stochastic calculus has an
        extra second-derivative term. Informally:
        """
    )
    st.markdown(r"$$dW_t^2 = dt$$")

    st.subheader("Stochastic Integrals")
    st.markdown(
        """
        A stochastic integral adds up random increments weighted by information
        known at the time of each increment:
        """
    )
    st.markdown(r"$$\int_0^T H_t\,dW_t \approx \sum_i H_{t_i}(W_{t_{i+1}}-W_{t_i})$$")
    st.markdown(
        """
        The integrand $H_t$ must be adaptive: at time $t_i$, it can use
        information already observed, but not the future Brownian increment.
        This is what makes the integral compatible with modelling sequential
        uncertainty.

        When $H_t$ is well behaved, the stochastic integral with respect to
        Brownian motion has no drift by itself. It can move up or down, but its
        expected increment is zero.
        """
    )

    st.subheader("Ito Processes")
    st.markdown(
        """
        An Ito process separates predictable drift from Brownian noise:
        """
    )
    st.markdown(r"$$dX_t = a_t\,dt + b_t\,dW_t$$")
    st.markdown(
        """
        The coefficient $a_t$ controls the local average change. The coefficient
        $b_t$ controls the local random shock size. In finance, this is the
        basic language for writing continuous-time models.
        """
    )
    st.markdown(
        """
        The scale rules are not arbitrary. Imagine cutting a time horizon $T$
        into $N$ equal pieces, so each piece has length:
        """
    )
    st.markdown(r"$$dt=\frac{T}{N}$$")
    st.markdown(
        """
        A deterministic drift step proportional to $dt$ accumulates to a finite
        nonzero amount:
        """
    )
    st.markdown(r"$$\sum_{i=1}^{N}\mu\,dt=N\mu\frac{T}{N}=\mu T$$")
    st.markdown(
        """
        But a per-step term of size $dt^2$ disappears when the grid is refined:
        """
    )
    st.markdown(r"$$\sum_{i=1}^{N}dt^2=N\left(\frac{T}{N}\right)^2=\frac{T^2}{N}\to0$$")
    st.markdown(
        """
        A mixed term $dt\\,dW_t$ also disappears. Since a Brownian increment has
        typical size $\\sqrt{dt}$, the product has size $dt^{3/2}$:
        """
    )
    st.markdown(
        r"$$\sum_{i=1}^{N}dt\,\Delta W_i"
        r"\text{ has scale }"
        r"Ndt^{3/2}=N\left(\frac{T}{N}\right)^{3/2}"
        r"=\frac{T^{3/2}}{\sqrt{N}}\to0$$"
    )
    st.markdown(
        """
        Brownian increments are different. A single Brownian increment has
        typical size $\\sqrt{dt}$. The signed increments do not add like a drift;
        their random signs partly cancel. But the squared increments are
        positive, so their total size is:
        """
    )
    st.markdown(r"$$\sum_{i=1}^{N}(\Delta W_i)^2\approx Ndt=T$$")
    st.markdown(
        """
        That is the construction behind the symbolic rules. We keep terms that
        accumulate to a finite contribution over time. We drop terms whose total
        contribution vanishes.
        """
    )
    st.table(
        [
            {
                "Rule": "$dt^2=0$",
                "Scale": "$dt^2$",
                "Why it vanishes or survives": (
                    "Smaller than order $dt$, so it disappears in the limit."
                ),
            },
            {
                "Rule": "$dt\\,dW_t=0$",
                "Scale": "$dt\\sqrt{dt}=dt^{3/2}$",
                "Why it vanishes or survives": "Also smaller than order $dt$, so it disappears.",
            },
            {
                "Rule": "$dW_t^2=dt$",
                "Scale": "$(\\sqrt{dt})^2=dt$",
                "Why it vanishes or survives": (
                    "Order $dt$, so Brownian quadratic variation survives."
                ),
            },
        ]
    )
    st.markdown(
        """
        This is why stochastic calculus keeps a second-order term. In ordinary
        calculus, all squared differential terms are too small to matter. With
        Brownian motion, the square of the random increment is exactly the size
        that still contributes to the drift.
        """
    )

    st.subheader("Ito's Lemma")
    st.markdown(
        """
        Ito's lemma is the stochastic version of the chain rule. The ordinary
        chain rule says that if $x(t)$ is smooth and $y(t)=f(t,x(t))$, then:
        """
    )
    st.markdown(
        "$$"
        r"dy=\frac{\partial f}{\partial t}dt"
        r"+\frac{\partial f}{\partial x}dx"
        "$$"
    )
    st.markdown(
        """
        For an Ito process, the input is not smooth:
        """
    )
    st.markdown(r"$$dX_t=a_t\,dt+b_t\,dW_t$$")
    st.markdown(
        """
        A second-order Taylor expansion gives:
        """
    )
    st.markdown(
        "$$"
        r"df\approx "
        r"\frac{\partial f}{\partial t}dt"
        r"+\frac{\partial f}{\partial x}dX_t"
        r"+\frac{1}{2}\frac{\partial^2 f}{\partial x^2}(dX_t)^2"
        "$$"
    )
    st.markdown(
        """
        Now expand $(dX_t)^2$ using the Ito rules:
        """
    )
    st.markdown(
        "$$"
        r"(dX_t)^2=(a_tdt+b_tdW_t)^2"
        r"=a_t^2dt^2+2a_tb_tdt\,dW_t+b_t^2dW_t^2"
        r"=b_t^2dt"
        "$$"
    )
    st.markdown(
        """
        Substituting this back into the Taylor expansion gives Ito's lemma:
        """
    )
    st.markdown(
        "$$"
        r"df(t,X_t)=\left("
        r"\frac{\partial f}{\partial t}"
        r"+a_t\frac{\partial f}{\partial x}"
        r"+\frac{1}{2}b_t^2\frac{\partial^2 f}{\partial x^2}"
        r"\right)dt"
        r"+b_t\frac{\partial f}{\partial x}dW_t"
        "$$"
    )
    st.markdown(
        """
        Compared with the ordinary chain rule, the new term is:
        """
    )
    st.markdown(r"$$\frac{1}{2}b_t^2\frac{\partial^2 f}{\partial x^2}dt$$")
    st.markdown(
        """
        The new term measures curvature exposure to randomness. If $f$ is
        linear, then $f_{xx}=0$ and Ito's lemma reduces to the ordinary chain
        rule. If $f$ is curved, symmetric random moves do not cancel perfectly:
        convex functions gain from variance, concave functions lose from it.
        This is the same intuition behind gamma in option pricing.
        """
    )

    st.subheader("Using Ito's Lemma to Integrate")
    st.markdown(
        """
        Ito's lemma is often used in reverse. If applying the lemma creates a
        term we want to integrate, we can rearrange the equation.

        Take:
        """
    )
    st.markdown(r"$$f(W_t)=\frac{1}{2}W_t^2$$")
    st.markdown(
        """
        Here the input process is just Brownian motion, so $dX_t=dW_t$,
        $a_t=0$, and $b_t=1$. The derivatives are:
        """
    )
    st.markdown(
        r"$$\frac{\partial f}{\partial x}=W_t,\qquad "
        r"\frac{\partial^2 f}{\partial x^2}=1$$"
    )
    st.markdown(
        """
        Ito's lemma gives:
        """
    )
    st.markdown(r"$$d\left(\frac{1}{2}W_t^2\right)=W_t\,dW_t+\frac{1}{2}dt$$")
    st.markdown(
        """
        Now rearrange and integrate from $0$ to $t$:
        """
    )
    st.markdown(r"$$W_t\,dW_t=d\left(\frac{1}{2}W_t^2\right)-\frac{1}{2}dt$$")
    st.markdown(r"$$\int_0^t W_s\,dW_s=\frac{1}{2}W_t^2-\frac{1}{2}t$$")
    st.markdown(
        """
        So the chart below is not just a numerical trick. It is showing a
        stochastic integral evaluated by applying Ito's lemma and rearranging
        the result.
        """
    )

    st.subheader("Experiment")
    left, right = st.columns(2)

    with left:
        maturity = st.number_input(
            "Time horizon",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.25,
            key="stochastic_calculus_maturity",
        )
        n_steps = st.number_input(
            "Path steps",
            min_value=20,
            max_value=2_000,
            value=500,
            step=50,
            key="stochastic_calculus_steps",
        )

    with right:
        seed = st.number_input(
            "Seed",
            min_value=0,
            value=42,
            step=1,
            key="stochastic_calculus_seed",
        )

    increments = brownian_increments(
        n_paths=1,
        n_steps=int(n_steps),
        dt=float(maturity) / int(n_steps),
        seed=int(seed),
    )[0]
    time_grid = np.linspace(0.0, float(maturity), int(n_steps) + 1)
    brownian_path = np.concatenate(([0.0], np.cumsum(increments)))

    st.markdown(
        """
        Smooth path versus Brownian path:

        The smooth path is a deterministic log curve, $\\log(1+t)$. It is not a
        financial model; it is just a familiar smooth curve to compare with
        Brownian motion.

        The comparison is about squared increments, not raw increments. For a
        smooth path, $\\sum_i (x_{t_{i+1}}-x_{t_i})^2$ goes to zero as the grid
        is refined. For Brownian motion,
        $\\sum_i (W_{t_{i+1}}-W_{t_i})^2$ accumulates to elapsed time.
        """
    )
    roughness_rows = _roughness_rows(time_grid=time_grid, brownian_path=brownian_path)
    roughness_values = [float(row["value"]) for row in roughness_rows]
    roughness_min, roughness_max = padded_range(roughness_values)
    st.vega_lite_chart(
        roughness_rows,
        line_chart_spec(
            x_field="time",
            x_title="Time",
            y_field="value",
            y_title="Path value",
            y_min=roughness_min,
            y_max=roughness_max,
        ),
        use_container_width=True,
    )

    st.markdown(
        """
        Quadratic variation:

        The smooth path's quadratic variation shrinks toward zero. This is not
        because sine is periodic or because it ends where it started; any
        sufficiently smooth path has zero quadratic variation. Brownian
        quadratic variation remains close to the time horizon because its
        increments have typical size $\\sqrt{dt}$.
        """
    )
    qv_rows = _quadratic_variation_rows(maturity=float(maturity), seed=int(seed))
    qv_values = [float(row["quadratic_variation"]) for row in qv_rows]
    qv_min, qv_max = padded_range([*qv_values, 0.0, float(maturity)])
    st.vega_lite_chart(
        qv_rows,
        _quadratic_variation_spec(y_min=qv_min, y_max=qv_max),
        use_container_width=True,
    )

    st.markdown(
        """
        Stochastic integral example:

        For the integral $\\int_0^t W_s\\,dW_s$, the ordinary chain-rule guess
        would be $\\frac{1}{2}W_t^2$. Ito's correction subtracts
        $\\frac{1}{2}t$:
        """
    )
    st.markdown(r"$$\int_0^t W_s\,dW_s = \frac{1}{2}W_t^2-\frac{1}{2}t$$")
    integral_rows = _stochastic_integral_rows(time_grid=time_grid, brownian_path=brownian_path)
    integral_values = [float(row["value"]) for row in integral_rows]
    integral_min, integral_max = padded_range(integral_values)
    st.vega_lite_chart(
        integral_rows,
        line_chart_spec(
            x_field="time",
            x_title="Time",
            y_field="value",
            y_title="Value",
            y_min=integral_min,
            y_max=integral_max,
        ),
        use_container_width=True,
    )


def _roughness_rows(*, time_grid: np.ndarray, brownian_path: np.ndarray) -> list[ChartRow]:
    rows: list[ChartRow] = []
    smooth_path = np.log1p(time_grid)
    scaled_brownian_path = brownian_path / max(float(np.std(brownian_path)), 1e-12)
    stride = max(1, len(time_grid) // 300)

    for time, smooth_value, brownian_value in zip(
        time_grid[::stride],
        smooth_path[::stride],
        scaled_brownian_path[::stride],
        strict=True,
    ):
        rows.append({"time": float(time), "series": "Smooth path", "value": float(smooth_value)})
        rows.append(
            {
                "time": float(time),
                "series": "Brownian path (scaled)",
                "value": float(brownian_value),
            }
        )

    return rows


def _quadratic_variation_rows(*, maturity: float, seed: int) -> list[ChartRow]:
    rows: list[ChartRow] = []
    for n_steps in [8, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 5_000]:
        time_grid = np.linspace(0.0, maturity, n_steps + 1)
        smooth_path = np.log1p(time_grid)
        smooth_increments = np.diff(smooth_path)
        brownian_path_increments = brownian_increments(
            n_paths=1,
            n_steps=n_steps,
            dt=maturity / n_steps,
            seed=seed,
        )[0]

        rows.append(
            {
                "steps": n_steps,
                "series": "Smooth path",
                "quadratic_variation": float(np.sum(smooth_increments**2)),
            }
        )
        rows.append(
            {
                "steps": n_steps,
                "series": "Brownian path",
                "quadratic_variation": float(np.sum(brownian_path_increments**2)),
            }
        )
        rows.append(
            {
                "steps": n_steps,
                "series": "Time horizon",
                "quadratic_variation": maturity,
            }
        )

    return rows


def _quadratic_variation_spec(*, y_min: float, y_max: float) -> ChartSpec:
    return {
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {
                "field": "steps",
                "type": "quantitative",
                "title": "Time steps",
                "scale": {"type": "log"},
            },
            "y": {
                "field": "quadratic_variation",
                "type": "quantitative",
                "title": "Sum of squared increments",
                "scale": {"domain": [y_min, y_max], "zero": False},
            },
            "color": {"field": "series", "type": "nominal", "title": ""},
        },
    }


def _stochastic_integral_rows(
    *,
    time_grid: np.ndarray,
    brownian_path: np.ndarray,
) -> list[ChartRow]:
    increments = np.diff(brownian_path)
    left_endpoint_values = brownian_path[:-1]
    integral_values = np.concatenate(([0.0], np.cumsum(left_endpoint_values * increments)))
    ordinary_chain_rule_values = 0.5 * brownian_path**2
    ito_values = ordinary_chain_rule_values - 0.5 * time_grid

    rows: list[ChartRow] = []
    stride = max(1, len(time_grid) // 300)
    for time, integral_value, ordinary_value, ito_value in zip(
        time_grid[::stride],
        integral_values[::stride],
        ordinary_chain_rule_values[::stride],
        ito_values[::stride],
        strict=True,
    ):
        rows.append(
            {
                "time": float(time),
                "series": "Stochastic integral sum",
                "value": float(integral_value),
            }
        )
        rows.append(
            {
                "time": float(time),
                "series": "Ordinary chain-rule guess",
                "value": float(ordinary_value),
            }
        )
        rows.append(
            {
                "time": float(time),
                "series": "Ito-corrected value",
                "value": float(ito_value),
            }
        )

    return rows

from __future__ import annotations

import numpy as np
from components.charts import ChartRow, ChartSpec, line_chart_spec, padded_range
from components.source import show_source_file

import streamlit as st
from quant_lab.stochastic.brownian import brownian_increments


def render() -> None:
    st.header("Probability Primer")
    st.caption("Random variables, processes, and the stochastic calculus intuition we need.")

    st.markdown(
        """
        Pricing models do not predict one future price. They describe a
        distribution of possible future prices. That is why the next labs need a
        small amount of probability language before we derive geometric Brownian
        motion or Black-Scholes.
        """
    )

    st.subheader("Random Variables")
    st.markdown(
        """
        A random variable is a number whose value is uncertain before we observe
        it. For example, tomorrow's stock return is a random variable in a
        model. Its distribution describes which values are possible and how
        likely they are.

        A useful first example is a standard normal random variable:
        """
    )
    st.markdown(r"$$Z \sim \mathcal{N}(0,1)$$")
    st.markdown(
        """
        This says that $Z$ has mean $0$ and variance $1$. In simulations we draw
        many samples from this distribution. The average of those samples will
        not be exactly zero, but it should get closer as the sample size grows.
        """
    )
    st.table(
        [
            {
                "Concept": "Expectation",
                "Notation": "$E[X]$",
                "Meaning": "Long-run average value of a random variable.",
            },
            {
                "Concept": "Variance",
                "Notation": "$Var(X)$",
                "Meaning": "Average squared distance from the expectation.",
            },
            {
                "Concept": "Standard deviation",
                "Notation": "$\\sqrt{Var(X)}$",
                "Meaning": "Typical scale of variation in the same units as X.",
            },
            {
                "Concept": "Standard error",
                "Notation": "$SE(\\bar{X})$",
                "Meaning": "Uncertainty of a sample average.",
            },
        ]
    )

    st.subheader("Random Processes")
    st.markdown(
        """
        A random process is a collection of random variables indexed by time.
        Instead of one uncertain number $X$, we have a whole path:
        """
    )
    st.markdown(r"$$X_0, X_{\Delta t}, X_{2\Delta t}, \ldots, X_T$$")
    st.markdown(
        """
        A stock model is a random process because the price can evolve through
        many possible paths before reaching maturity. Later, $S_t$ will denote
        the stock price at time $t$.
        """
    )

    st.subheader("Brownian Motion")
    st.markdown(
        """
        Brownian motion is the basic random process used to inject continuous
        noise into many finance models. It is also called a Wiener process,
        named after Norbert Wiener, which is why we write it as $W_t$.

        It starts at zero, has independent increments, and each increment is
        normally distributed with variance equal to the length of the time
        interval:
        """
    )
    st.markdown(r"$$W_{t+\Delta t}-W_t \sim \mathcal{N}(0,\Delta t)$$")
    st.markdown(
        """
        Independent increments means that the next shock does not remember the
        previous shock. The variance scaling means that over a small time step
        $\\Delta t$, the shock has typical size $\\sqrt{\\Delta t}$.

        The $\\Delta t$ in the variance is part of the standard mathematical
        definition of Brownian motion. It is not specific to quantitative
        finance. If the elapsed time doubles, the variance doubles:
        """
    )
    st.markdown(r"$$W_{t+2\Delta t}-W_t \sim \mathcal{N}(0,2\Delta t)$$")
    st.markdown(
        """
        So uncertainty grows linearly in variance, while the typical move grows
        like the square root of time.
        """
    )

    st.subheader("Stochastic Differential Notation")
    st.markdown(
        """
        In ordinary calculus, $dx$ represents a tiny deterministic change. In
        stochastic calculus, $dW_t$ represents a tiny random Brownian increment.
        A stochastic differential equation often separates predictable drift
        from random noise:
        """
    )
    st.markdown(r"$$dX_t = \mu\,dt + \sigma\,dW_t$$")
    st.markdown(
        r"""
        Here $\mu\,dt$ is the drift term: the average directional change over a
        tiny time step. The term $\sigma\,dW_t$ is the random shock. The
        parameter $\sigma$ controls the shock size and is called volatility in
        finance.
        """
    )

    st.subheader("Why $dW_t^2 = dt$ Appears")
    st.markdown("The most unfamiliar stochastic calculus rule is:")
    st.markdown(r"$$dW_t^2 = dt$$")
    st.markdown(
        """
        This is not ordinary algebra. It is shorthand for a limiting fact:
        Brownian increments are of size $\\sqrt{dt}$, so their squares are of
        size $dt$. When many squared increments are added along a path, they
        accumulate to the elapsed time. This accumulated sum is called quadratic
        variation.
        """
    )
    st.markdown(r"$$\sum_i (W_{t_{i+1}}-W_{t_i})^2 \rightarrow T$$")

    st.subheader("Implementation")
    show_source_file("src/quant_lab/stochastic/brownian.py")

    st.subheader("Experiment")
    left, right = st.columns(2)

    with left:
        n_samples = st.number_input(
            "Normal samples",
            min_value=100,
            max_value=100_000,
            value=5_000,
            step=1_000,
        )
        n_paths = st.number_input(
            "Brownian paths",
            min_value=1,
            max_value=50,
            value=12,
            step=1,
        )

    with right:
        maturity = st.number_input(
            "Time horizon",
            min_value=0.01,
            max_value=5.0,
            value=1.0,
            step=0.25,
        )
        n_steps = st.number_input(
            "Time steps",
            min_value=5,
            max_value=1_000,
            value=252,
            step=25,
        )
        seed = st.number_input("Seed", min_value=0, value=42, step=1)

    normal_samples = _normal_samples(n_samples=int(n_samples), seed=int(seed))
    sample_mean = float(np.mean(normal_samples))
    sample_std = float(np.std(normal_samples, ddof=1))
    sample_se = sample_std / float(np.sqrt(len(normal_samples)))

    metric_left, metric_middle, metric_right = st.columns(3)
    metric_left.metric("Sample mean", f"{sample_mean:.4f}")
    metric_middle.metric("Sample std", f"{sample_std:.4f}")
    metric_right.metric("Std error of mean", f"{sample_se:.4f}")

    st.markdown(
        """
        Standard normal samples:

        The histogram is a simulation view of one random variable. Increasing
        the sample count should make the shape more stable and the sample mean
        closer to zero.
        """
    )
    st.vega_lite_chart(_normal_rows(normal_samples), _histogram_spec(), use_container_width=True)

    increments = brownian_increments(
        n_paths=int(n_paths),
        n_steps=int(n_steps),
        dt=float(maturity) / int(n_steps),
        seed=int(seed),
    )
    paths = np.concatenate(
        (np.zeros((int(n_paths), 1), dtype=np.float64), np.cumsum(increments, axis=1)),
        axis=1,
    )
    time_grid = np.linspace(0.0, float(maturity), int(n_steps) + 1)

    st.markdown(
        """
        Brownian paths:

        Each line is one possible path of the same random process. The paths
        wander, but the distribution of increments is controlled by the time
        step.
        """
    )
    path_rows = _path_rows(paths, time_grid)
    path_values = [float(row["value"]) for row in path_rows]
    y_min, y_max = padded_range(path_values)
    st.vega_lite_chart(
        path_rows,
        line_chart_spec(
            x_field="time",
            x_title="Time",
            y_field="value",
            y_title="Brownian value",
            y_min=y_min,
            y_max=y_max,
            color_field="path",
        ),
        use_container_width=True,
    )

    st.markdown(
        """
        Quadratic variation:

        For a Brownian path over horizon $T$, the sum of squared increments
        tends to $T$ as the time grid becomes finer. This is the numerical
        version of the intuition behind $dW_t^2=dt$.
        """
    )
    variation_rows = _quadratic_variation_rows(maturity=float(maturity), seed=int(seed))
    variation_values = [float(row["quadratic_variation"]) for row in variation_rows]
    variation_min, variation_max = padded_range([*variation_values, float(maturity)])
    st.vega_lite_chart(
        variation_rows,
        _quadratic_variation_spec(y_min=variation_min, y_max=variation_max),
        use_container_width=True,
    )


def _normal_samples(*, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(rng.standard_normal(size=n_samples), dtype=np.float64)


def _normal_rows(samples: np.ndarray) -> list[ChartRow]:
    return [{"sample": float(sample)} for sample in samples]


def _histogram_spec() -> ChartSpec:
    return {
        "mark": "bar",
        "encoding": {
            "x": {
                "field": "sample",
                "type": "quantitative",
                "bin": {"maxbins": 50},
                "title": "Sample value",
            },
            "y": {"aggregate": "count", "type": "quantitative", "title": "Count"},
        },
    }


def _path_rows(paths: np.ndarray, time_grid: np.ndarray) -> list[ChartRow]:
    rows: list[ChartRow] = []
    stride = max(1, len(time_grid) // 250)
    for path_index, path in enumerate(paths):
        for time, value in zip(time_grid[::stride], path[::stride], strict=True):
            rows.append(
                {
                    "path": f"Path {path_index + 1}",
                    "time": round(float(time), 6),
                    "value": float(value),
                }
            )
    return rows


def _quadratic_variation_rows(*, maturity: float, seed: int) -> list[ChartRow]:
    rows: list[ChartRow] = []
    for n_steps in [8, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 5_000]:
        increments = brownian_increments(
            n_paths=1,
            n_steps=n_steps,
            dt=maturity / n_steps,
            seed=seed,
        )
        quadratic_variation = float(np.sum(increments[0] ** 2))
        rows.append(
            {
                "steps": n_steps,
                "series": "Simulated quadratic variation",
                "quadratic_variation": quadratic_variation,
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

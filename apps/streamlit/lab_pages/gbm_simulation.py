from __future__ import annotations

from math import exp

import numpy as np
from components.charts import ChartRow, ChartSpec, line_chart_spec, padded_range
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
        Black-Scholes world. The Probability Primer introduces the Brownian
        motion shocks used here. GBM keeps prices positive by modeling
        proportional returns instead of absolute price changes.
        """
    )

    st.subheader("Concept")
    st.markdown(
        """
        A simple additive model would say that the next price is today's price
        plus a random change. That is easy to write down, but it can produce
        negative stock prices. Equity prices are better thought of in
        proportional terms: a move from 100 to 101 and a move from 200 to 202
        are both one percent moves.

        GBM models proportional returns. In continuous time, the percentage
        change in price has a drift part and a Brownian shock part.

        Symbols used below:

        - $S_t$: underlying price at time $t$
        - $\\mu$: expected continuously compounded return in the physical model
        - $r$: continuously compounded risk-free rate
        - $q$: continuously compounded dividend yield
        - $\\sigma$: annualized volatility
        - $W_t$: standard Brownian motion, introduced in the Probability Primer
        - $\\Delta t$: one simulation time step
        - $Z$: a standard normal random variable
        """
    )

    st.subheader("Derivation")
    st.markdown(
        """
        Start with the idea that price changes should scale with the current
        price. A deterministic proportional model would be:
        """
    )
    st.markdown(r"$$\frac{dS_t}{S_t}=\mu\,dt$$")
    st.markdown(
        """
        To make the future uncertain, add Brownian noise to the proportional
        return:
        """
    )
    st.markdown(r"$$\frac{dS_t}{S_t}=\mu\,dt+\sigma\,dW_t$$")
    st.markdown(r"$$dS_t=\mu S_t\,dt+\sigma S_t\,dW_t$$")
    st.markdown(
        """
        This is the continuous-time shorthand for geometric Brownian motion.
        But for simulation, it is clearer to think in terms of log returns.
        The log return over one time step is:
        """
    )
    st.markdown(r"$$\log(S_{t+\Delta t})-\log(S_t)$$")
    st.markdown(
        """
        Log returns are useful because they add over time. If a price moves from
        $100$ to $110$ and later from $110$ to $121$, the simple returns are
        both $10\\%$, while the log returns add to the total log return from
        $100$ to $121$.

        GBM assumes that log returns over a time step are normally distributed:
        """
    )
    st.table(
        [
            {
                "Move": "100 -> 110",
                "Simple return": "10.00%",
                "Log return": "log(110 / 100)",
            },
            {
                "Move": "110 -> 121",
                "Simple return": "10.00%",
                "Log return": "log(121 / 110)",
            },
            {
                "Move": "100 -> 121",
                "Simple return": "21.00%",
                "Log return": "log(121 / 100)",
            },
        ]
    )
    st.markdown(
        "$$"
        r"\log S_{t+\Delta t}-\log S_t = "
        r"(\mu-\frac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}Z"
        "$$"
    )
    st.markdown(
        "$$"
        r"S_{t+\Delta t} = S_t \exp\left("
        r"(\mu-\frac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}Z"
        r"\right)"
        "$$"
    )
    st.markdown(
        """
        This formula explains two key properties.

        First, prices stay positive. The exponential term is always strictly
        positive, even when the drift or the random shock is very negative. A
        very negative exponent can make the next price very small, but it cannot
        make it negative:
        """
    )
    st.markdown(r"$$S_t>0,\quad e^x>0\quad\Rightarrow\quad S_{t+\Delta t}>0$$")
    st.markdown(
        """
        Second, terminal prices are lognormal. The log of the terminal price is
        normally distributed because it is the initial log price plus normal log
        returns. If $\\log(S_T)$ is normal, then $S_T$ is lognormal:
        """
    )
    st.markdown(
        "$$"
        r"\log(S_T) \sim \mathcal{N}\left("
        r"\log(S_0)+(\mu-\frac{1}{2}\sigma^2)T,\ \sigma^2T"
        r"\right)"
        "$$"
    )
    st.markdown(
        """
        The $-\\frac{1}{2}\\sigma^2$ term is the stochastic-calculus correction
        that makes the log-price equation consistent with the proportional-price
        equation. A later Black-Scholes derivation will introduce Ito's lemma,
        which is the formal rule behind this correction. For now, the important
        practical point is that the simulator evolves log returns and then
        exponentiates them back into prices.
        """
    )
    st.markdown(
        """
        In the pricing labs we usually simulate under the risk-neutral model,
        where the drift is $r-q$ instead of an estimated real-world return
        $\\mu$:
        """
    )
    st.markdown(
        "$$"
        r"S_{t+\Delta t} = S_t \exp\left("
        r"(r-q-\frac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}Z"
        r"\right)"
        "$$"
    )

    st.subheader("Why This Model Is Used")
    st.markdown(
        """
        GBM is not a perfect model of real markets, but it is a useful baseline:

        - prices stay positive
        - returns compound naturally
        - terminal prices are lognormally distributed
        - the model leads to the Black-Scholes closed form
        - it is easy to simulate and test

        Its limitations matter too. GBM assumes constant volatility, independent
        returns, continuous paths, and normally distributed log returns. Real
        markets show volatility clustering, jumps, heavy tails, transaction
        costs, and changing liquidity. We use GBM first because it is the clean
        reference model, not because it is the final word.
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

    st.markdown(
        """
        Sample paths:

        Each line is one possible price history from the same model. Increasing
        volatility should widen the fan of paths. Changing the rate or dividend
        yield changes the average drift, but individual paths can still move in
        either direction.
        """
    )
    sample_path_count = min(25, int(n_paths))
    time_grid = np.linspace(0.0, float(maturity), int(n_steps) + 1)
    path_rows = _path_rows(paths[:sample_path_count], time_grid)
    path_values = [float(row["price"]) for row in path_rows]
    path_min, path_max = padded_range(path_values)
    st.vega_lite_chart(
        path_rows,
        line_chart_spec(
            x_field="time",
            x_title="Time",
            y_field="price",
            y_title="Simulated price",
            y_min=path_min,
            y_max=path_max,
            color_field="path",
        ),
        use_container_width=True,
    )

    st.markdown(
        """
        Terminal price distribution:

        The paths are noisy one by one, but their terminal prices form a stable
        distribution as the number of paths grows. Under GBM, this distribution
        is lognormal: prices stay positive and the right tail can stretch
        farther than the left.
        """
    )
    st.vega_lite_chart(
        _terminal_price_rows(terminal_prices),
        _histogram_spec(),
        use_container_width=True,
    )

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

    st.markdown(
        """
        The simulated terminal mean should be close to the risk-neutral expected
        terminal mean when the number of paths is large. They will not match
        exactly because simulation uses a finite sample.
        """
    )


def _path_rows(paths: np.ndarray, time_grid: np.ndarray) -> list[ChartRow]:
    rows: list[ChartRow] = []
    stride = max(1, len(time_grid) // 250)
    for path_index, path in enumerate(paths):
        for time, price in zip(time_grid[::stride], path[::stride], strict=True):
            rows.append(
                {
                    "path": f"Path {path_index + 1}",
                    "time": round(float(time), 6),
                    "price": float(price),
                }
            )
    return rows


def _terminal_price_rows(terminal_prices: np.ndarray) -> list[ChartRow]:
    return [{"terminal_price": float(price)} for price in terminal_prices]


def _histogram_spec() -> ChartSpec:
    return {
        "mark": "bar",
        "encoding": {
            "x": {
                "field": "terminal_price",
                "type": "quantitative",
                "bin": {"maxbins": 60},
                "title": "Terminal price",
            },
            "y": {"aggregate": "count", "type": "quantitative", "title": "Count"},
        },
    }

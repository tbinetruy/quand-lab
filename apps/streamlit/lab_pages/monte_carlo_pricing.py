from __future__ import annotations

from components.source import show_source_file

import streamlit as st
from quant_lab.domain import BlackScholesMarket, EuropeanOption, MonteCarloConfig, OptionType
from quant_lab.pricing import black_scholes_price, monte_carlo_price


def render() -> None:
    st.header("Monte Carlo Pricing")
    st.caption("Estimate European option prices by averaging discounted simulated payoffs.")

    st.markdown(
        """
        Monte Carlo pricing turns the risk-neutral pricing equation into a
        simulation experiment. We simulate many possible terminal prices,
        evaluate the option payoff in each scenario, average those payoffs, and
        discount back to today.
        """
    )

    st.subheader("Concept")
    st.markdown(
        """
        Symbols used below:

        - $V_0$: option value today
        - $r$: continuously compounded risk-free rate
        - $T$: maturity in years
        - $S_T$: simulated terminal underlying price
        - $\\Phi(S_T)$: option payoff at maturity
        - $N$: number of simulated paths
        - $\\hat{V}_0$: Monte Carlo estimate of the option value
        """
    )

    st.subheader("Math")
    st.markdown(r"$$V_0 = e^{-rT} E^Q[\Phi(S_T)]$$")
    st.markdown(
        r"""
        $$\hat{V}_0 = e^{-rT}\frac{1}{N}\sum_{i=1}^{N}\Phi(S_T^{(i)})$$
        """
    )
    st.markdown(
        "$$"
        r"\operatorname{SE}(\hat{V}_0) = "
        r"\frac{\operatorname{std}(e^{-rT}\Phi(S_T))}{\sqrt{N}}"
        "$$"
    )

    st.subheader("Implementation")
    payoff_tab, pricer_tab = st.tabs(["Payoffs", "Monte Carlo pricer"])

    with payoff_tab:
        show_source_file("src/quant_lab/pricing/payoffs.py")

    with pricer_tab:
        show_source_file("src/quant_lab/pricing/monte_carlo.py")

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
        maturity = st.number_input("Maturity in years", min_value=0.0, value=1.0, step=0.25)
        spot = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0)

    with right:
        rate = st.number_input("Risk-free rate", value=0.05, step=0.01, format="%.4f")
        dividend_yield = st.number_input("Dividend yield", value=0.0, step=0.01, format="%.4f")
        volatility = st.number_input("Volatility", min_value=0.0, value=0.2, step=0.05)
        n_paths = st.number_input(
            "Paths",
            min_value=100,
            max_value=100_000,
            value=20_000,
            step=1_000,
        )
        n_steps = st.number_input("Time steps", min_value=1, max_value=1_000, value=64, step=16)
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
        antithetic = st.toggle("Antithetic variates", value=True)

    option = EuropeanOption(
        option_type=option_type,
        strike=strike,
        maturity_years=maturity,
    )
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

    result = monte_carlo_price(option, market, config)
    analytical_price = black_scholes_price(option, market)
    lower_bound, upper_bound = result.confidence_interval_95

    metric_left, metric_middle, metric_right = st.columns(3)
    metric_left.metric("Monte Carlo price", f"{result.price:.4f}")
    metric_middle.metric("Black-Scholes price", f"{analytical_price:.4f}")
    metric_right.metric("Absolute error", f"{abs(result.price - analytical_price):.4f}")

    st.json(
        {
            "standard_error": round(result.standard_error, 6),
            "confidence_interval_95": [round(lower_bound, 6), round(upper_bound, 6)],
            "black_scholes_inside_interval": lower_bound <= analytical_price <= upper_bound,
        }
    )

    if result.paths is not None:
        st.markdown("Sample simulated paths:")
        sample_path_count = min(25, n_paths)
        st.line_chart(result.paths[:sample_path_count].T)

    st.markdown("Convergence against Black-Scholes:")
    convergence_counts = _convergence_counts(n_paths)
    convergence_prices = [
        monte_carlo_price(
            option,
            market,
            MonteCarloConfig(
                n_paths=path_count,
                n_steps=n_steps,
                seed=seed,
                antithetic=antithetic,
                return_paths=False,
            ),
        ).price
        for path_count in convergence_counts
    ]
    convergence_rows = [
        {
            "paths": path_count,
            "series": "Monte Carlo",
            "price": convergence_price,
        }
        for path_count, convergence_price in zip(
            convergence_counts,
            convergence_prices,
            strict=True,
        )
    ]
    convergence_rows.extend(
        {
            "paths": path_count,
            "series": "Black-Scholes",
            "price": analytical_price,
        }
        for path_count in convergence_counts
    )
    price_y_min, price_y_max = _padded_range([*convergence_prices, analytical_price])
    st.vega_lite_chart(
        convergence_rows,
        _line_chart_spec(
            y_field="price",
            y_title="Price",
            y_min=price_y_min,
            y_max=price_y_max,
        ),
        use_container_width=True,
    )

    st.markdown("Pricing error:")
    pricing_errors = [price - analytical_price for price in convergence_prices]
    error_y_min, error_y_max = _padded_range([*pricing_errors, 0.0])
    st.vega_lite_chart(
        [
            {
                "paths": path_count,
                "series": "Pricing error",
                "error": pricing_error,
            }
            for path_count, pricing_error in zip(
                convergence_counts,
                pricing_errors,
                strict=True,
            )
        ],
        _line_chart_spec(
            y_field="error",
            y_title="Monte Carlo - Black-Scholes",
            y_min=error_y_min,
            y_max=error_y_max,
        ),
        use_container_width=True,
    )
    st.caption("Convergence points use increasing path counts from the selected seed.")


def _convergence_counts(max_path_count: int) -> list[int]:
    candidates = [100, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000]
    counts = [candidate for candidate in candidates if candidate <= max_path_count]

    if counts and counts[-1] == max_path_count:
        return counts

    return [*counts, max_path_count]


def _padded_range(values: list[float]) -> tuple[float, float]:
    lower_bound = min(values)
    upper_bound = max(values)
    span = upper_bound - lower_bound

    if span == 0.0:
        padding = max(abs(lower_bound) * 0.05, 1.0)
    else:
        padding = span * 0.1

    return lower_bound - padding, upper_bound + padding


def _line_chart_spec(
    *,
    y_field: str,
    y_title: str,
    y_min: float,
    y_max: float,
) -> dict[str, object]:
    return {
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {"field": "paths", "type": "quantitative", "title": "Paths"},
            "y": {
                "field": y_field,
                "type": "quantitative",
                "title": y_title,
                "scale": {"domain": [y_min, y_max], "zero": False},
            },
            "color": {"field": "series", "type": "nominal", "title": ""},
        },
    }

from __future__ import annotations

from math import exp

from components.charts import ChartRow, line_chart_spec, linspace, padded_range, surface_chart_spec
from components.source import show_source_file

import streamlit as st
from quant_lab.domain import BlackScholesMarket, EuropeanOption, OptionType
from quant_lab.risk.greeks import black_scholes_greeks
from quant_lab.risk.portfolio import (
    OptionPosition,
    PortfolioPosition,
    PositionValuation,
    StockPosition,
    estimate_portfolio_value_change,
    option_portfolio_payoff,
    value_option_portfolio,
)


def render() -> None:
    st.header("Applied Option Risk")
    st.caption("Use prices, Greeks, and scenarios together on a small option portfolio.")

    st.markdown(
        """
        The earlier labs priced one option at a time. Real risk work usually
        starts after that: combine positions, aggregate exposures, and ask what
        happens when spot, volatility, or time moves. This page applies the
        Black-Scholes and Greeks machinery to a small synthetic strategy before
        we introduce live market data. If "long" and "short" are new terms,
        the Foundations page defines them before the strategy examples here.
        """
    )

    st.subheader("Concept")
    st.markdown(
        """
        A portfolio value is the signed sum of its position values. A long
        option contributes positive value and positive Greeks. A short option
        contributes the opposite. Stock contributes value and delta, while cash
        contributes value but no spot, volatility, or time sensitivity in this
        simplified setup.

        This is the first place where Greeks become operational. They are not
        just properties of one formula anymore: they become a compact summary
        of portfolio risk.
        """
    )
    st.markdown(
        """
        The key distinction is local versus full repricing:

        - Greeks estimate small moves around the current state.
        - Full repricing recomputes every position after a scenario shock.

        If the shock is small, the Greek approximation should usually be close.
        If the shock is large, option convexity, moneyness changes, and time
        effects make full repricing more reliable.
        """
    )
    st.markdown("Strategy vocabulary used in the experiment:")
    st.table(
        [
            {
                "Strategy": "Covered call",
                "Construction": "Long stock, short call",
                "Basic idea": "Keep stock exposure but sell away upside above the strike.",
            },
            {
                "Strategy": "Protective put",
                "Construction": "Long stock, long put",
                "Basic idea": "Keep stock upside while buying downside protection.",
            },
            {
                "Strategy": "Long straddle",
                "Construction": "Long call, long put, same strike",
                "Basic idea": "Pay premium to benefit from a large move in either direction.",
            },
            {
                "Strategy": "Bull call spread",
                "Construction": "Long lower-strike call, short higher-strike call",
                "Basic idea": "Express capped upside with lower cost than a single long call.",
            },
            {
                "Strategy": "Bear put spread",
                "Construction": "Long higher-strike put, short lower-strike put",
                "Basic idea": "Express capped downside with lower cost than a single long put.",
            },
            {
                "Strategy": "Delta-hedged call",
                "Construction": "Long call, short delta shares of stock",
                "Basic idea": "Remove first-order spot exposure at the current market state.",
            },
        ]
    )

    st.subheader("Math")
    st.markdown(
        """
        For positions indexed by $j$, with signed quantity $w_j$ and unit price
        $V_j$, portfolio value is:
        """
    )
    st.markdown(r"$$\Pi=\sum_j w_j V_j$$")
    st.markdown(
        """
        Since differentiation is linear, portfolio Greeks aggregate the same
        way:
        """
    )
    st.markdown(r"$$\Delta_\Pi=\sum_j w_j\Delta_j,\qquad \Gamma_\Pi=\sum_j w_j\Gamma_j$$")
    st.markdown(r"$$\nu_\Pi=\sum_j w_j\nu_j,\qquad \Theta_\Pi=\sum_j w_j\Theta_j$$")
    st.markdown(
        """
        The local P&L estimate comes from the same Taylor expansion used in
        ordinary calculus. If the portfolio value is a function of spot,
        $\\Pi(S)$, then after a small spot move from $S$ to $S+\\Delta S$:
        """
    )
    st.markdown(
        "$$"
        r"\Pi(S+\Delta S)-\Pi(S)\approx"
        r"\frac{\partial \Pi}{\partial S}\Delta S"
        r"+\frac{1}{2}\frac{\partial^2\Pi}{\partial S^2}(\Delta S)^2"
        "$$"
    )
    st.markdown(
        """
        Delta is the first derivative and gamma is the second derivative:
        """
    )
    st.markdown(
        "$$"
        r"\Delta_\Pi=\frac{\partial \Pi}{\partial S},"
        r"\qquad"
        r"\Gamma_\Pi=\frac{\partial^2\Pi}{\partial S^2}"
        "$$"
    )
    st.markdown(
        """
        Be careful with the notation: the symbol $\\Delta$ is doing two jobs.
        In $\\Delta S$, it means "change in spot." In $\\Delta_\\Pi$, it means
        "the portfolio delta Greek." Substituting those Greek names into the
        Taylor expansion gives:
        """
    )
    st.markdown(
        "$$"
        r"\underbrace{\Pi(S+\Delta S)-\Pi(S)}_{\text{portfolio value change}}"
        r"\approx "
        r"\underbrace{\Delta_\Pi \Delta S}_{\text{linear delta effect}}"
        r"+"
        r"\underbrace{\frac{1}{2}\Gamma_\Pi(\Delta S)^2}_{\text{curvature correction}}"
        "$$"
    )
    st.markdown(
        """
        In words: start with the delta estimate, then add a gamma correction
        because an option portfolio is usually curved rather than a straight
        line in spot.
        """
    )
    st.markdown(
        """
        The portfolio value is not only a function of spot. It also depends on
        volatility, time, and rates:
        """
    )
    st.markdown(r"$$\Pi=\Pi(S,\sigma,t,r)$$")
    st.markdown(
        """
        A multi-input Taylor approximation adds one local sensitivity term for
        each input. Keeping the second-order spot term gives:
        """
    )
    st.markdown(
        "$$"
        r"\Delta \Pi \approx "
        r"\underbrace{\Delta_\Pi \Delta S}_{\text{spot level}}"
        r"+"
        r"\underbrace{\frac{1}{2}\Gamma_\Pi(\Delta S)^2}_{\text{spot curvature}}"
        r"+"
        r"\underbrace{\nu_\Pi \Delta\sigma}_{\text{volatility shock}}"
        r"+"
        r"\underbrace{\Theta_\Pi \Delta t}_{\text{time passing}}"
        r"+"
        r"\underbrace{\rho_\Pi \Delta r}_{\text{rate shock}}"
        "$$"
    )
    st.markdown(
        """
        This is still a local approximation. It is useful for small shocks and
        quick risk summaries. For larger shocks, the experiment below also does
        full repricing so we can see where the local approximation starts to
        drift away.
        """
    )

    st.subheader("Implementation")
    show_source_file("src/quant_lab/risk/portfolio.py")

    st.subheader("Experiment")
    left, right = st.columns(2)

    with left:
        strategy = st.selectbox(
            "Strategy",
            options=[
                "Covered call",
                "Protective put",
                "Long straddle",
                "Bull call spread",
                "Bear put spread",
                "Delta-hedged call",
            ],
        )
        spot = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0)
        strike = st.number_input("Main strike", min_value=0.01, value=100.0, step=1.0)
        strike_width = st.number_input("Spread width", min_value=0.01, value=10.0, step=1.0)

    with right:
        maturity = st.number_input("Maturity in years", min_value=0.01, value=1.0, step=0.25)
        rate = st.number_input("Risk-free rate", value=0.05, step=0.01, format="%.4f")
        dividend_yield = st.number_input("Dividend yield", value=0.0, step=0.01, format="%.4f")
        volatility = st.number_input("Volatility", min_value=0.01, value=0.2, step=0.05)

    market = BlackScholesMarket(
        spot=spot,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    positions = _strategy_positions(
        strategy=strategy,
        strike=strike,
        strike_width=strike_width,
        maturity=maturity,
        market=market,
    )
    valuation = value_option_portfolio(positions, market)

    st.markdown(_strategy_explanation(strategy))
    st.table(_position_rows(positions, valuation.positions))

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Portfolio value", f"{valuation.total_value:.4f}")
    metric_2.metric("Delta", f"{valuation.total_greeks.delta:.4f}")
    metric_3.metric("Gamma", f"{valuation.total_greeks.gamma:.4f}")
    metric_4.metric("Vega", f"{valuation.total_greeks.vega:.4f}")

    metric_5, metric_6, metric_7 = st.columns(3)
    metric_5.metric("Theta", f"{valuation.total_greeks.theta:.4f}")
    metric_6.metric("Rho", f"{valuation.total_greeks.rho:.4f}")
    metric_7.metric("Position count", str(len(positions)))

    payoff_spots = linspace(max(spot * 0.4, 0.01), spot * 1.8, 80)
    payoff_rows = _payoff_rows(
        positions=positions,
        terminal_spots=payoff_spots,
        initial_value=valuation.total_value,
        rate=rate,
        maturity=maturity,
    )
    payoff_values = [float(row["value"]) for row in payoff_rows]
    payoff_y_min, payoff_y_max = padded_range(payoff_values)

    st.markdown("Terminal payoff and P&L:")
    st.markdown(
        """
        The payoff line is the cashflow at maturity. The P&L line subtracts
        the future value of today's portfolio cost, so it answers a more useful
        question: did this strategy beat simply carrying its initial value at
        the risk-free rate?
        """
    )
    st.vega_lite_chart(
        payoff_rows,
        line_chart_spec(
            x_field="spot",
            x_title="Terminal spot",
            y_field="value",
            y_title="Cashflow",
            y_min=payoff_y_min,
            y_max=payoff_y_max,
        ),
        use_container_width=True,
    )

    scenario_rows = _spot_scenario_rows(
        positions=positions,
        market=market,
        initial_value=valuation.total_value,
    )
    scenario_values = [float(row["value"]) for row in scenario_rows]
    scenario_y_min, scenario_y_max = padded_range(scenario_values)

    st.markdown("Full repricing versus Greek approximation:")
    st.markdown(
        """
        Full repricing recalculates the whole portfolio at each spot. The local
        estimate uses today's aggregated delta and gamma. The gap between the
        lines is the cost of using a local approximation too far from the
        current state.
        """
    )
    st.vega_lite_chart(
        scenario_rows,
        line_chart_spec(
            x_field="spot",
            x_title="Scenario spot",
            y_field="value",
            y_title="P&L from current value",
            y_min=scenario_y_min,
            y_max=scenario_y_max,
        ),
        use_container_width=True,
    )

    st.markdown("Portfolio value surface over spot and volatility:")
    st.markdown(
        """
        This surface shows full repricing, not a Greek approximation. Use it to
        see how the same strategy can carry different risks in different
        regions of spot and volatility.
        """
    )
    surface_rows = _value_surface_rows(positions=positions, market=market)
    surface_values = [float(row["price"]) for row in surface_rows]
    st.vega_lite_chart(
        surface_rows,
        surface_chart_spec(
            value_title="Portfolio value",
            value_min=min(surface_values),
            value_max=max(surface_values),
            value_field="price",
        ),
        use_container_width=True,
    )


def _strategy_positions(
    *,
    strategy: str,
    strike: float,
    strike_width: float,
    maturity: float,
    market: BlackScholesMarket,
) -> tuple[PortfolioPosition, ...]:
    call = EuropeanOption(OptionType.CALL, strike=strike, maturity_years=maturity)
    put = EuropeanOption(OptionType.PUT, strike=strike, maturity_years=maturity)
    high_call = EuropeanOption(
        OptionType.CALL,
        strike=strike + strike_width,
        maturity_years=maturity,
    )
    low_put = EuropeanOption(
        OptionType.PUT,
        strike=max(strike - strike_width, 0.01),
        maturity_years=maturity,
    )

    if strategy == "Covered call":
        return (StockPosition(quantity=1.0), OptionPosition(call, quantity=-1.0))
    if strategy == "Protective put":
        return (StockPosition(quantity=1.0), OptionPosition(put, quantity=1.0))
    if strategy == "Long straddle":
        return (OptionPosition(call, quantity=1.0), OptionPosition(put, quantity=1.0))
    if strategy == "Bull call spread":
        return (
            OptionPosition(call, quantity=1.0),
            OptionPosition(high_call, quantity=-1.0),
        )
    if strategy == "Bear put spread":
        return (
            OptionPosition(put, quantity=1.0),
            OptionPosition(low_put, quantity=-1.0),
        )

    call_delta = black_scholes_greeks(call, market).delta
    return (
        OptionPosition(call, quantity=1.0),
        StockPosition(quantity=-call_delta),
    )


def _position_rows(
    positions: tuple[PortfolioPosition, ...],
    valuations: tuple[PositionValuation, ...],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    for position, valuation in zip(positions, valuations, strict=True):
        rows.append(
            {
                "position": _position_name(position),
                "quantity": _position_quantity(position),
                "unit price": valuation.unit_price,
                "market value": valuation.market_value,
                "delta": valuation.total_greeks.delta,
                "gamma": valuation.total_greeks.gamma,
                "vega": valuation.total_greeks.vega,
            }
        )

    return rows


def _payoff_rows(
    *,
    positions: tuple[PortfolioPosition, ...],
    terminal_spots: list[float],
    initial_value: float,
    rate: float,
    maturity: float,
) -> list[ChartRow]:
    rows: list[ChartRow] = []
    future_cost = initial_value * exp(rate * maturity)

    for terminal_spot in terminal_spots:
        payoff = option_portfolio_payoff(positions, terminal_spot)
        rows.append({"spot": terminal_spot, "value": payoff, "series": "Payoff"})
        rows.append({"spot": terminal_spot, "value": payoff - future_cost, "series": "P&L"})

    return rows


def _spot_scenario_rows(
    *,
    positions: tuple[PortfolioPosition, ...],
    market: BlackScholesMarket,
    initial_value: float,
) -> list[ChartRow]:
    rows: list[ChartRow] = []
    valuation = value_option_portfolio(positions, market)
    scenario_spots = linspace(max(market.spot * 0.5, 0.01), market.spot * 1.5, 80)

    for scenario_spot in scenario_spots:
        scenario_market = BlackScholesMarket(
            spot=scenario_spot,
            rate=market.rate,
            volatility=market.volatility,
            dividend_yield=market.dividend_yield,
        )
        scenario_value = value_option_portfolio(positions, scenario_market).total_value
        spot_change = scenario_spot - market.spot
        estimated_change = estimate_portfolio_value_change(valuation, spot_change=spot_change)

        rows.append(
            {
                "spot": scenario_spot,
                "value": scenario_value - initial_value,
                "series": "Full reprice",
            }
        )
        rows.append(
            {
                "spot": scenario_spot,
                "value": estimated_change,
                "series": "Delta-gamma estimate",
            }
        )

    return rows


def _value_surface_rows(
    *,
    positions: tuple[PortfolioPosition, ...],
    market: BlackScholesMarket,
) -> list[ChartRow]:
    rows: list[ChartRow] = []
    spots = linspace(max(market.spot * 0.6, 0.01), market.spot * 1.4, 26)
    volatilities = linspace(max(market.volatility * 0.4, 0.01), market.volatility * 1.8, 20)

    for volatility in volatilities:
        for spot in spots:
            scenario_market = BlackScholesMarket(
                spot=spot,
                rate=market.rate,
                volatility=volatility,
                dividend_yield=market.dividend_yield,
            )
            rows.append(
                {
                    "spot": round(spot, 2),
                    "volatility": round(volatility, 4),
                    "price": value_option_portfolio(positions, scenario_market).total_value,
                    "option_type": "Portfolio",
                }
            )

    return rows


def _position_name(position: PortfolioPosition) -> str:
    if isinstance(position, OptionPosition):
        option_type = position.option.option_type.value.title()
        return f"{option_type} K={position.option.strike:g}"
    if isinstance(position, StockPosition):
        return "Stock"

    return "Cash"


def _position_quantity(position: PortfolioPosition) -> float:
    if isinstance(position, OptionPosition):
        return position.quantity * position.contract_multiplier
    if isinstance(position, StockPosition):
        return position.quantity

    return position.amount


def _strategy_explanation(strategy: str) -> str:
    explanations = {
        "Covered call": (
            "A covered call holds the stock and sells a call. It keeps stock "
            "downside, collects option premium, and gives up upside above the strike."
        ),
        "Protective put": (
            "A protective put holds the stock and buys a put. It keeps upside "
            "while paying for downside protection below the strike."
        ),
        "Long straddle": (
            "A long straddle buys a call and a put at the same strike. It is "
            "long volatility and benefits from large moves in either direction."
        ),
        "Bull call spread": (
            "A bull call spread buys a lower-strike call and sells a higher-strike "
            "call. It is a capped upside position with lower upfront cost than a call."
        ),
        "Bear put spread": (
            "A bear put spread buys a higher-strike put and sells a lower-strike "
            "put. It is a capped downside position with lower upfront cost than a put."
        ),
        "Delta-hedged call": (
            "A delta-hedged call buys the call and shorts stock equal to the "
            "call's current delta. It starts near delta-neutral, but gamma means "
            "the hedge changes as spot moves."
        ),
    }
    return explanations[strategy]

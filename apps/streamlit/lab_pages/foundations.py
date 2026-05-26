from __future__ import annotations

from math import exp

from components.charts import ChartRow, float_array, line_chart_spec, linspace, padded_range

import streamlit as st
from quant_lab.domain import BlackScholesMarket, EuropeanOption, OptionType
from quant_lab.pricing import black_scholes_price, european_payoff


def render() -> None:
    st.header("Foundations")
    st.caption("Options vocabulary before the pricing models.")

    st.markdown(
        """
        An option is a contract whose value depends on an underlying asset, such
        as a stock. The first labs focus on European options, which can be
        exercised only at maturity.

        The buyer of an option has a right, not an obligation. The seller has
        the corresponding obligation if the buyer exercises that right.

        - A call option gives its buyer the right to buy the underlying at the
          strike price.
        - A put option gives its buyer the right to sell the underlying at the
          strike price.

        This is why calls are associated with upside and puts with downside. If
        you own a call, higher underlying prices can help you because you still
        have the right to buy at the fixed strike. If you own a put, lower
        underlying prices can help you because you still have the right to sell
        at the fixed strike.
        """
    )

    st.markdown("Buyer and seller roles:")
    st.table(
        [
            {
                "Option": "Call",
                "Buyer has the right to": "Buy the underlying at K",
                "Seller may be required to": "Sell the underlying at K",
            },
            {
                "Option": "Put",
                "Buyer has the right to": "Sell the underlying at K",
                "Seller may be required to": "Buy the underlying at K",
            },
        ]
    )
    st.markdown(
        """
        Market language usually shortens "buyer" and "seller" into long and
        short:

        - long an option: you bought the option, so you own its payoff
        - short an option: you sold or wrote the option, so you owe its payoff

        Long and short describe the sign of the position, not whether the
        option is a call or a put. You can be long a call, short a call, long a
        put, or short a put.
        """
    )
    st.table(
        [
            {
                "Position": "Long call",
                "Meaning": "Bought a call",
                "Payoff exposure": "Benefits from upside above K",
            },
            {
                "Position": "Short call",
                "Meaning": "Sold a call",
                "Payoff exposure": "Owes upside above K",
            },
            {
                "Position": "Long put",
                "Meaning": "Bought a put",
                "Payoff exposure": "Benefits from downside below K",
            },
            {
                "Position": "Short put",
                "Meaning": "Sold a put",
                "Payoff exposure": "Owes downside below K",
            },
        ]
    )

    st.subheader("Vocabulary")
    st.markdown(
        """
        We will use these symbols throughout the lab:

        - $S_0$: current spot price of the underlying asset
        - $S_T$: underlying price at maturity
        - $K$: strike price, the contractual exercise price
        - $T$: maturity in years
        - $r$: continuously compounded risk-free rate
        - $q$: continuously compounded dividend yield
        - $\\sigma$: annualized volatility
        - $\\Phi(S_T)$: payoff at maturity
        - $V_0$: option price today
        """
    )
    st.markdown(
        """
        The price paid for an option is often called the premium. In this lab we
        usually say price because we are computing theoretical prices from
        models rather than quoting a broker screen.
        """
    )
    st.markdown(
        """
        Moneyness describes where spot is relative to strike:

        - in-the-money: exercising would produce positive payoff
        - at-the-money: spot is close to strike
        - out-of-the-money: exercising would produce zero payoff

        This is not just vocabulary. Many shapes in later charts concentrate
        near the strike because that is where moneyness can change with a small
        move in the underlying.
        """
    )

    st.subheader("Payoff")
    st.markdown(
        """
        The payoff is the cashflow at maturity. It is not the same thing as the
        option price today. Payoff depends only on the terminal underlying price
        and the strike. Price today also depends on time, volatility, rates, and
        the probability of different terminal prices.
        """
    )
    st.markdown(r"$$\Phi_{call}(S_T)=\max(S_T-K,0)$$")
    st.markdown(r"$$\Phi_{put}(S_T)=\max(K-S_T,0)$$")
    st.markdown(
        """
        A call payoff is flat until the underlying finishes above the strike,
        then increases one-for-one with the underlying. A put payoff is the
        mirror image: it is valuable when the underlying finishes below the
        strike.

        Example with $K=100$: a call pays $20$ if $S_T=120$ and zero if
        $S_T=80$. A put pays $20$ if $S_T=80$ and zero if $S_T=120$.
        """
    )

    st.subheader("Price, Intrinsic Value, and Time Value")
    st.markdown(
        """
        Intrinsic value is what the option would be worth if it expired
        immediately. Time value is the extra value from uncertainty that remains
        before maturity.
        """
    )
    st.markdown(r"$$V_0 = \text{intrinsic value} + \text{time value}$$")
    st.markdown(
        """
        A one-year at-the-money option can have large time value because the
        future payoff is still uncertain. Deep in-the-money options are mostly
        intrinsic value. Deep out-of-the-money options may have low value, but
        they can still be worth something if there is enough time and
        volatility.
        """
    )

    st.subheader("Model Price Preview")
    st.markdown(
        """
        The experiment below shows a Black-Scholes model price. A model price is
        not a market quote. It is the price implied by a set of modelling
        assumptions: European exercise, geometric Brownian motion, constant
        volatility, constant interest rate, and no transaction costs.
        """
    )
    st.markdown(
        """
        The risk-free rate $r$ is the continuously compounded return used to
        discount future cashflows in the model. The idea is not that any real
        trade is perfectly risk-free. It is a mathematical benchmark: a cash
        amount payable at maturity is worth less today because money can earn
        interest over time.
        """
    )
    st.markdown(
        """
        Later pages will introduce risk-neutral pricing. The notation
        $\\mathbb{E}^{Q}$ means "expectation under the risk-neutral probability
        measure $Q$." For now, read the pricing idea as:
        """
    )
    st.markdown(r"$$\text{price today}=\text{discounted model average of future payoff}$$")
    st.markdown(
        """
        We delay the formal equation until the probability and Black-Scholes
        pages, where $Q$ and no-arbitrage can be defined properly.
        """
    )

    st.subheader("Experiment")
    left, right = st.columns(2)

    with left:
        selected_option_type = st.segmented_control(
            "Option type",
            options=["call", "put"],
            format_func=lambda value: value.title(),
            default="call",
            key="foundations_option_type",
        )
        option_type = OptionType(selected_option_type or "call")
        spot = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0)
        strike = st.number_input("Strike", min_value=0.01, value=100.0, step=1.0)

    with right:
        maturity = st.number_input("Maturity in years", min_value=0.0, value=1.0, step=0.25)
        rate = st.number_input("Risk-free rate", value=0.05, step=0.01, format="%.4f")
        dividend_yield = st.number_input("Dividend yield", value=0.0, step=0.01, format="%.4f")
        volatility = st.number_input("Volatility", min_value=0.0, value=0.2, step=0.05)

    option = EuropeanOption(option_type=option_type, strike=strike, maturity_years=maturity)
    market = BlackScholesMarket(
        spot=spot,
        rate=rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
    )
    price = black_scholes_price(option, market)
    intrinsic_value = _intrinsic_value(option, spot)
    time_value = price - intrinsic_value

    metric_left, metric_middle, metric_right = st.columns(3)
    metric_left.metric("Black-Scholes price", f"{price:.4f}")
    metric_middle.metric("Intrinsic value", f"{intrinsic_value:.4f}")
    metric_right.metric("Time value", f"{time_value:.4f}")

    st.markdown(
        """
        The discounted values below are present-value versions of the strike and
        spot cashflows used by Black-Scholes. They will matter more once we
        derive the closed-form formula.
        """
    )
    st.json(
        {
            "moneyness": _moneyness(option, spot),
                "discounted_strike": round(strike * exp(-rate * maturity), 6),
                "discounted_spot": round(spot * exp(-dividend_yield * maturity), 6),
            }
    )

    st.markdown("Payoff at maturity:")
    st.table(_payoff_scenarios(option))
    payoff_chart_rows = _payoff_rows(option)
    payoff_values = [float(row["payoff"]) for row in payoff_chart_rows]
    payoff_min, payoff_max = padded_range(payoff_values)
    st.vega_lite_chart(
        payoff_chart_rows,
        line_chart_spec(
            x_field="terminal_spot",
            x_title="Terminal spot",
            y_field="payoff",
            y_title="Payoff",
            y_min=payoff_min,
            y_max=payoff_max,
            color_field=None,
        ),
        use_container_width=True,
    )

    st.markdown(
        """
        Price versus intrinsic value:

        The gap between the two lines is time value. It is usually largest near
        the strike because a small future move can change whether the option
        finishes in-the-money.
        """
    )
    value_rows = _value_rows(option, market)
    value_min, value_max = padded_range([float(row["value"]) for row in value_rows])
    st.vega_lite_chart(
        value_rows,
        line_chart_spec(
            x_field="spot",
            x_title="Spot today",
            y_field="value",
            y_title="Value",
            y_min=value_min,
            y_max=value_max,
        ),
        use_container_width=True,
    )


def _payoff_rows(option: EuropeanOption) -> list[ChartRow]:
    spots = linspace(max(0.01, option.strike * 0.5), option.strike * 1.5, 120)
    payoffs = european_payoff(option, float_array(spots))
    return [
        {
            "terminal_spot": round(spot, 4),
            "payoff": float(payoff),
            "option_type": option.option_type.value.title(),
        }
        for spot, payoff in zip(spots, payoffs, strict=True)
    ]


def _payoff_scenarios(option: EuropeanOption) -> list[dict[str, float | str]]:
    terminal_spots = [
        option.strike * 0.8,
        option.strike,
        option.strike * 1.2,
    ]
    payoffs = european_payoff(option, float_array(terminal_spots))
    return [
        {
            "Terminal spot": round(terminal_spot, 2),
            "Relation to strike": _strike_relation(terminal_spot, option.strike),
            "Payoff": round(float(payoff), 2),
        }
        for terminal_spot, payoff in zip(terminal_spots, payoffs, strict=True)
    ]


def _value_rows(option: EuropeanOption, market: BlackScholesMarket) -> list[ChartRow]:
    rows: list[ChartRow] = []
    spot_min = max(0.01, option.strike * 0.5)
    spot_max = option.strike * 1.5

    for surface_spot in linspace(spot_min, spot_max, 120):
        surface_market = BlackScholesMarket(
            spot=surface_spot,
            rate=market.rate,
            volatility=market.volatility,
            dividend_yield=market.dividend_yield,
        )
        comparison_option = EuropeanOption(
            option.option_type,
            strike=option.strike,
            maturity_years=option.maturity_years,
        )
        price = black_scholes_price(comparison_option, surface_market)
        intrinsic_value = _intrinsic_value(comparison_option, surface_spot)
        rows.append(
            {
                "spot": round(surface_spot, 4),
                "series": "Black-Scholes price",
                "value": price,
                "option_type": option.option_type.value.title(),
            }
        )
        rows.append(
            {
                "spot": round(surface_spot, 4),
                "series": "Intrinsic value",
                "value": intrinsic_value,
                "option_type": option.option_type.value.title(),
            }
        )

    return rows


def _intrinsic_value(option: EuropeanOption, spot: float) -> float:
    if option.option_type is OptionType.CALL:
        return max(spot - option.strike, 0.0)
    return max(option.strike - spot, 0.0)


def _moneyness(option: EuropeanOption, spot: float) -> str:
    relative_distance = abs(spot - option.strike) / option.strike
    if relative_distance < 0.02:
        return "at-the-money"
    if option.option_type is OptionType.CALL:
        return "in-the-money" if spot > option.strike else "out-of-the-money"
    return "in-the-money" if spot < option.strike else "out-of-the-money"


def _strike_relation(spot: float, strike: float) -> str:
    if spot < strike:
        return "Below strike"
    if spot > strike:
        return "Above strike"
    return "At strike"

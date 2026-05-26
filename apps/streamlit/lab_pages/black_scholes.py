from __future__ import annotations

from math import exp

from components.charts import (
    ChartRow,
    line_chart_spec,
    linspace,
    padded_range,
    payoff_chart_spec,
    payoff_rows,
    surface_chart_spec,
    surface_value_max,
    surface_value_min,
)
from components.source import show_source_file

import streamlit as st
from quant_lab.domain import BlackScholesMarket, EuropeanOption, OptionType
from quant_lab.pricing import black_scholes_price


def render() -> None:
    st.header("Black-Scholes Closed Form")
    st.caption("A first analytical benchmark for European option prices.")

    st.markdown(
        """
        Black-Scholes gives a closed-form price for European calls and puts
        under a specific set of assumptions. This is the benchmark we will use
        when we later check Monte Carlo convergence and finite-difference
        solvers. If calls, puts, moneyness, or time value are new, start with
        the Foundations page first. If Ito's lemma is new, read the Stochastic
        Calculus page before this derivation.
        """
    )

    st.subheader("Concept")
    st.markdown(
        """
        Black-Scholes starts with a model for the underlying asset and derives
        the option price from a hedging argument. The key idea is no-arbitrage:
        if two portfolios have the same future cashflows, they should have the
        same price today.

        Main assumptions:

        - the underlying follows geometric Brownian motion
        - volatility is constant
        - the risk-free rate and dividend yield are constant
        - trading is continuous and frictionless
        - the option is European, so exercise happens only at maturity
        - the underlying can be bought, sold, and used for hedging

        Symbols used below:

        - $S_0$: current spot price
        - $S_t$: underlying price at time $t$
        - $K$: option strike
        - $T$: maturity in years
        - $V(S,t)$: option value when spot is $S$ at time $t$
        - $r$: continuously compounded risk-free rate
        - $q$: continuously compounded dividend yield
        - $\\sigma$: annualized volatility
        - $\\Delta$: hedge ratio, later called delta
        - $N(x)$: cumulative distribution function of the standard normal distribution
        """
    )

    st.subheader("Derivation")
    st.markdown(
        """
        Start with a real-world GBM model for the underlying:
        """
    )
    st.markdown(r"$$dS_t=\mu S_t\,dt+\sigma S_t\,dW_t$$")
    st.markdown(
        """
        The option value is a function of time and spot: $V(t,S_t)$. Apply
        Ito's lemma to this function. Here:
        """
    )
    st.markdown(
        "$$"
        r"a_t=\mu S_t,\qquad b_t=\sigma S_t"
        "$$"
    )
    st.markdown(
        """
        Substituting into Ito's lemma gives:
        """
    )
    st.markdown(
        "$$"
        r"dV=\left(\frac{\partial V}{\partial t}"
        r"+\mu S\frac{\partial V}{\partial S}"
        r"+\frac{1}{2}\sigma^2S^2\frac{\partial^2V}{\partial S^2}\right)dt"
        r"+\sigma S\frac{\partial V}{\partial S}dW_t"
        "$$"
    )
    st.markdown(
        """
        The term multiplied by $dW_t$ is the local randomness in the option
        value. Black-Scholes constructs a portfolio that removes this
        randomness instant by instant:
        """
    )
    st.markdown(r"$$\Pi = V - \Delta S$$")
    st.markdown(
        """
        This means: hold one option and short $\\Delta$ shares of the underlying.
        Over a tiny time step, the self-financing change in the portfolio is:
        """
    )
    st.markdown(r"$$d\Pi=dV-\Delta\,dS-\Delta qS\,dt$$")
    st.markdown(
        """
        The last term is the dividend cashflow. Because the portfolio is short
        $\\Delta$ shares, it must pay the dividends on those shares. That cash
        outflow is $-\\Delta qSdt$.
        """
    )
    st.markdown(
        """
        Now substitute $dV$ and $dS$:
        """
    )
    st.markdown(
        "$$"
        r"d\Pi="
        r"\left(\frac{\partial V}{\partial t}"
        r"+\mu S\frac{\partial V}{\partial S}"
        r"+\frac{1}{2}\sigma^2S^2\frac{\partial^2V}{\partial S^2}\right)dt"
        r"+\sigma S\frac{\partial V}{\partial S}dW_t"
        r"-\Delta(\mu Sdt+\sigma SdW_t)"
        r"-\Delta qSdt"
        "$$"
    )
    st.markdown(
        """
        Group the random $dW_t$ terms:
        """
    )
    st.markdown(
        "$$"
        r"\sigma S\left(\frac{\partial V}{\partial S}-\Delta\right)dW_t"
        "$$"
    )
    st.markdown(
        """
        Choose the hedge ratio:
        """
    )
    st.markdown(r"$$\Delta=\frac{\partial V}{\partial S}$$")
    st.markdown(
        """
        This cancels the Brownian shock. The portfolio is locally riskless, so
        no-arbitrage says it must earn the risk-free rate:
        """
    )
    st.markdown(r"$$d\Pi=r\Pi\,dt=r\left(V-\Delta S\right)dt$$")
    st.markdown(
        """
        After setting $\\Delta=\\frac{\\partial V}{\\partial S}$, the left-hand
        side becomes:
        """
    )
    st.markdown(
        "$$"
        r"d\Pi="
        r"\left("
        r"\frac{\partial V}{\partial t}"
        r"+\frac{1}{2}\sigma^2S^2\frac{\partial^2V}{\partial S^2}"
        r"-qS\frac{\partial V}{\partial S}"
        r"\right)dt"
        "$$"
    )
    st.markdown(
        """
        Equate the two riskless returns:
        """
    )
    st.markdown(
        "$$"
        r"\frac{\partial V}{\partial t}"
        r"+\frac{1}{2}\sigma^2S^2\frac{\partial^2V}{\partial S^2}"
        r"-qS\frac{\partial V}{\partial S}"
        r"=r\left(V-S\frac{\partial V}{\partial S}\right)"
        "$$"
    )
    st.markdown(
        """
        Move everything to the left. This is the Black-Scholes PDE for a
        European option on an underlying with continuous dividend yield:
        """
    )
    st.markdown(
        "$$"
        r"\underbrace{\frac{\partial V}{\partial t}}_{\text{time change}}"
        r"+"
        r"\underbrace{\frac{1}{2}\sigma^2S^2\frac{\partial^2 V}{\partial S^2}}"
        r"_{\text{convexity / gamma term}}"
        r"+"
        r"\underbrace{(r-q)S\frac{\partial V}{\partial S}}"
        r"_{\text{risk-neutral drift term}}"
        r"-"
        r"\underbrace{rV}_{\text{discounting term}}"
        r"=0"
        "$$"
    )
    st.markdown(
        """
        Notice that $\\mu$ disappeared. The option price does not use the
        investor's expected stock return directly. The hedge removes the local
        Brownian risk, and the remaining riskless portfolio is priced from the
        risk-free rate.
        """
    )

    st.subheader("Closed Form")
    st.markdown(
        """
        The PDE route and the risk-neutral expectation route are equivalent in
        this model. For the closed form, the expectation route is more direct.
        We write $C$ for the call price today and $P$ for the put price today.

        Under the risk-neutral measure, the GBM terminal price is:
        """
    )
    st.markdown(
        "$$"
        r"S_T=S_0\exp\left("
        r"(r-q-\frac{1}{2}\sigma^2)T+\sigma\sqrt{T}Z"
        r"\right),\qquad Z\sim\mathcal{N}(0,1)"
        "$$"
    )
    st.markdown(
        """
        A European call pays only when $S_T>K$. Its payoff at maturity is:
        """
    )
    st.markdown(r"$$\Phi_{call}(S_T)=\max(S_T-K,0)$$")
    st.markdown(
        """
        The notation $(x)^+$ means the positive part of $x$:
        """
    )
    st.markdown(r"$$(x)^+=\max(x,0)$$")
    st.markdown(
        """
        Risk-neutral pricing says today's price is the discounted expected
        payoff under the pricing measure $Q$. Therefore:
        """
    )
    st.markdown(r"$$C=e^{-rT}\mathbb{E}^{Q}[(S_T-K)^+]$$")
    st.markdown(
        """
        Split the payoff into two pieces: receive the stock if exercise happens,
        and pay the strike if exercise happens.
        """
    )
    st.markdown(
        "$$"
        r"(S_T-K)^+=(S_T-K)\mathbf{1}_{S_T>K}"
        r"=S_T\mathbf{1}_{S_T>K}-K\mathbf{1}_{S_T>K}"
        "$$"
    )
    st.markdown(
        """
        The indicator $\\mathbf{1}_{S_T>K}$ is a random variable. Its
        expectation is the probability of the event:
        """
    )
    st.markdown(r"$$\mathbb{E}^{Q}[\mathbf{1}_{S_T>K}]=Q(S_T>K)$$")
    st.markdown(
        "$$"
        r"C=e^{-rT}\left("
        r"\mathbb{E}^{Q}[S_T\mathbf{1}_{S_T>K}]"
        r"-KQ(S_T>K)"
        r"\right)"
        "$$"
    )
    st.markdown(
        """
        The exercise condition can be written as a threshold on the standard
        normal variable $Z$. Start from the terminal GBM expression:
        """
    )
    st.markdown(
        "$$"
        r"S_0\exp\left((r-q-\frac{1}{2}\sigma^2)T+\sigma\sqrt{T}Z\right)>K"
        "$$"
    )
    st.markdown("Divide by $S_0$ and take logs:")
    st.markdown(
        "$$"
        r"(r-q-\frac{1}{2}\sigma^2)T+\sigma\sqrt{T}Z>\ln(K/S_0)"
        "$$"
    )
    st.markdown("Now isolate $Z$:")
    st.markdown(
        "$$"
        r"Z>"
        r"\frac{\ln(K/S_0)-(r-q-\frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}"
        "$$"
    )
    st.markdown(
        """
        Define $d_2$ as the negative of that threshold:
        """
    )
    st.markdown(
        "$$"
        r"d_2="
        r"\frac{\ln(S_0/K)+(r-q-\frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}"
        "$$"
    )
    st.markdown(
        """
        This definition is not arbitrary. It measures the risk-neutral expected
        log-moneyness at maturity in units of terminal log-volatility. With this
        definition, the exercise event is:
        """
    )
    st.markdown(r"$$S_T>K\Longleftrightarrow Z>-d_2$$")
    st.markdown(
        """
        By symmetry of the standard normal distribution:
        """
    )
    st.markdown(r"$$Q(S_T>K)=N(d_2)$$")
    st.markdown("The strike part is now clear:")
    st.markdown(r"$$e^{-rT}KQ(S_T>K)=Ke^{-rT}N(d_2)$$")
    st.markdown(
        """
        The stock part is a weighted expectation. The useful normal identity is:
        """
    )
    st.markdown(r"$$\operatorname{E}[e^{aZ}\mathbf{1}_{Z>c}]=e^{a^2/2}N(a-c)$$")
    st.markdown(
        """
        It comes from completing the square in the normal density:
        """
    )
    st.markdown(
        "$$"
        r"\int_c^\infty e^{az}\frac{1}{\sqrt{2\pi}}e^{-z^2/2}dz"
        r"=e^{a^2/2}\int_c^\infty"
        r"\frac{1}{\sqrt{2\pi}}e^{-(z-a)^2/2}dz"
        r"=e^{a^2/2}N(a-c)"
        "$$"
    )
    st.markdown(
        """
        Here $a=\\sigma\\sqrt{T}$ and $c=-d_2$, so $a-c=d_2+\\sigma\\sqrt{T}$.
        Define:
        """
    )
    st.markdown(r"$$d_1=d_2+\sigma\sqrt{T}$$")
    st.markdown(
        "$$"
        r"d_1="
        r"\frac{\ln(S_0/K)+(r-q+\frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}"
        "$$"
    )
    st.markdown("Then the discounted stock-weighted term becomes:")
    st.markdown(
        r"$$e^{-rT}\mathbb{E}^{Q}[S_T\mathbf{1}_{S_T>K}]"
        r"=S_0e^{-qT}N(d_1)$$"
    )
    st.markdown("Putting the stock and strike pieces together gives the call formula:")
    st.markdown(
        "$$"
        r"C="
        r"\underbrace{S_0e^{-qT}N(d_1)}_{\text{discounted stock leg}}"
        r"-"
        r"\underbrace{Ke^{-rT}N(d_2)}_{\text{discounted strike leg}}"
        "$$"
    )
    st.markdown(
        """
        The put formula follows from the same split, but now the payoff is
        positive when $S_T<K$:
        """
    )
    st.markdown(
        "$$"
        r"P=e^{-rT}\left("
        r"KQ(S_T<K)-\mathbb{E}^{Q}[S_T\mathbf{1}_{S_T<K}]"
        r"\right)"
        "$$"
    )
    st.markdown(
        """
        Since $Q(S_T<K)=N(-d_2)$ and the complementary stock-weighted term gives
        $S_0e^{-qT}N(-d_1)$:
        """
    )
    st.markdown(
        "$$"
        r"P="
        r"\underbrace{Ke^{-rT}N(-d_2)}_{\text{discounted strike leg}}"
        r"-"
        r"\underbrace{S_0e^{-qT}N(-d_1)}_{\text{discounted stock leg}}"
        "$$"
    )
    st.markdown(
        """
        So $d_2$ is tied to the exercise probability under the risk-neutral
        distribution, while $d_1$ appears because the stock part of the payoff
        is weighted by $S_T$ itself.
        """
    )
    st.markdown(
        """
        Put-call parity is a useful consistency check. A call minus a put with
        the same strike and maturity has the same payoff as a forward-like
        position, so the prices must satisfy:
        """
    )
    st.markdown(r"$$C-P=S_0e^{-qT}-Ke^{-rT}$$")

    st.subheader("Implementation")
    show_source_file("src/quant_lab/pricing/closed_form.py")

    st.subheader("Experiment")
    left, right = st.columns(2)

    with left:
        selected_option_type = st.segmented_control(
            "Option type",
            options=["call", "put"],
            format_func=lambda value: value.title(),
            default="call",
            key="black_scholes_option_type",
        )
        option_type = OptionType(selected_option_type or "call")
        strike = st.number_input("Strike", min_value=0.01, value=100.0, step=1.0)
        maturity = st.number_input("Maturity in years", min_value=0.0, value=1.0, step=0.25)

    with right:
        spot = st.number_input("Spot", min_value=0.01, value=100.0, step=1.0)
        rate = st.number_input("Risk-free rate", value=0.05, step=0.01, format="%.4f")
        dividend_yield = st.number_input("Dividend yield", value=0.0, step=0.01, format="%.4f")
        volatility = st.number_input("Volatility", min_value=0.0, value=0.2, step=0.05)

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

    price = black_scholes_price(option, market)
    st.metric("Closed-form price", f"{price:.4f}")

    call = EuropeanOption(OptionType.CALL, strike=strike, maturity_years=maturity)
    put = EuropeanOption(OptionType.PUT, strike=strike, maturity_years=maturity)
    call_price = black_scholes_price(call, market)
    put_price = black_scholes_price(put, market)
    parity_left = call_price - put_price
    parity_right = spot * exp(-dividend_yield * maturity) - strike * exp(-rate * maturity)

    st.markdown("Put-call parity check:")
    st.json(
        {
            "call_price_minus_put_price": round(parity_left, 8),
            "discounted_spot_minus_discounted_strike": round(parity_right, 8),
            "absolute_error": round(abs(parity_left - parity_right), 12),
        }
    )

    st.markdown(
        """
        Payoff diagram:

        This is the terminal cashflow, not today's model price. The call payoff
        bends upward above the strike. The put payoff bends upward below the
        strike.
        """
    )
    payoff_spots = linspace(max(0.01, strike * 0.5), strike * 1.5, 80)
    st.vega_lite_chart(
        payoff_rows(
            option_type=option_type,
            payoff_spots=payoff_spots,
            strike=strike,
            maturity=maturity,
        ),
        payoff_chart_spec(),
        key=f"black-scholes-payoff-{option_type.value}",
        use_container_width=True,
    )

    st.markdown(
        """
        Price curve:

        The model price is smoother than the payoff because it includes time
        value. Around the strike, uncertainty about finishing in-the-money is
        most important, so the curve rounds off the payoff kink.
        """
    )
    price_curve_rows = _price_curve_rows(option, market)
    curve_min, curve_max = padded_range([float(row["price"]) for row in price_curve_rows])
    st.vega_lite_chart(
        price_curve_rows,
        line_chart_spec(
            x_field="spot",
            x_title="Spot",
            y_field="price",
            y_title="Closed-form price",
            y_min=curve_min,
            y_max=curve_max,
            color_field=None,
        ),
        key=f"black-scholes-price-curve-{option_type.value}",
        use_container_width=True,
    )

    st.markdown(
        """
        Closed-form price surface over spot and volatility:

        Calls generally become more valuable as spot rises; puts generally
        become more valuable as spot falls. Both calls and puts usually become
        more valuable as volatility rises because wider future outcomes increase
        the value of optionality.
        """
    )
    surface_rows = _closed_form_surface_rows(option, market)
    st.vega_lite_chart(
        surface_rows,
        surface_chart_spec(
            value_title="Closed-form price",
            value_min=surface_value_min(surface_rows),
            value_max=surface_value_max(surface_rows),
        ),
        key=f"black-scholes-surface-{option_type.value}",
        use_container_width=True,
    )


def _price_curve_rows(
    option: EuropeanOption,
    market: BlackScholesMarket,
) -> list[ChartRow]:
    spot_min = max(0.01, option.strike * 0.5)
    spot_max = option.strike * 1.5

    rows: list[ChartRow] = []
    for curve_spot in linspace(spot_min, spot_max, 120):
        curve_market = BlackScholesMarket(
            spot=curve_spot,
            rate=market.rate,
            volatility=market.volatility,
            dividend_yield=market.dividend_yield,
        )
        rows.append(
            {
                "spot": round(curve_spot, 4),
                "price": black_scholes_price(option, curve_market),
                "option_type": option.option_type.value.title(),
            }
        )

    return rows


def _closed_form_surface_rows(
    option: EuropeanOption,
    market: BlackScholesMarket,
) -> list[ChartRow]:
    spot_min = max(0.01, market.spot * 0.6)
    spot_max = market.spot * 1.4
    volatility_max = max(0.6, market.volatility * 2.0, 0.1)

    rows: list[ChartRow] = []
    for surface_spot in linspace(spot_min, spot_max, 18):
        for surface_volatility in linspace(0.01, volatility_max, 18):
            surface_market = BlackScholesMarket(
                spot=surface_spot,
                rate=market.rate,
                volatility=surface_volatility,
                dividend_yield=market.dividend_yield,
            )
            rows.append(
                {
                    "spot": round(surface_spot, 2),
                    "volatility": round(surface_volatility, 3),
                    "price": round(black_scholes_price(option, surface_market), 6),
                    "option_type": option.option_type.value.title(),
                }
            )

    return rows

# Quant Lab Implementation Plan

## Current Strategy

Start with a modular typed Python engine and a disposable Streamlit frontend. Keep the engine clean enough that future Django, React, HTMX, API, cron, or Celery integrations can call the same code.

Prefer correctness, clarity, tests, and visual learning over runtime performance.

## Proposed Repository Shape

```text
quant/
  pyproject.toml
  README.md

  src/
    quant_lab/
      domain/
        instruments.py
        market.py
        quotes.py

      stochastic/
        brownian.py
        gbm.py

      models/
        black_scholes.py

      pricing/
        closed_form.py
        monte_carlo.py
        finite_difference.py

      risk/
        greeks.py
        hedging.py

      data/
        providers.py
        repositories.py

      experiments/
        black_scholes_experiment.py
        monte_carlo_experiment.py

      visualization/
        option_charts.py
        simulation_charts.py

  apps/
    streamlit/
      main.py
      pages/
        black_scholes.py
        monte_carlo.py

  tests/
```

This shape can evolve, but the key constraint should remain: `src/quant_lab` must be UI-agnostic.

## Milestone 1: Project Skeleton

Status: implemented.

Create the base Python project.

Tasks:

- add `pyproject.toml`
- configure typed Python package under `src/`
- add dependencies:
  - `numpy`
  - `scipy`
  - `pandas`
  - `plotly`
  - `streamlit`
  - `pytest`
  - `mypy` or `pyright`
  - `ruff`
- add initial README with run commands
- add importable `quant_lab` package
- add basic test and lint commands

Acceptance criteria:

- package imports locally
- tests run
- formatting/lint command runs
- Streamlit app can start, even if minimal

## Milestone 2: Domain Model

Status: implemented.

Implement small immutable typed domain objects.

Core types:

- `OptionType`
- `EuropeanOption`
- `BlackScholesMarket`
- `MonteCarloConfig`
- `MonteCarloResult`

Design notes:

- prefer `dataclass(frozen=True)`
- validate obvious invalid values near construction or use
- keep objects small and explicit
- avoid framework-specific validation early unless there is a strong reason

Acceptance criteria:

- invalid option parameters are rejected
- domain types are covered by simple tests
- domain types are usable without any UI dependency

## Milestone 2.5: Guided Streamlit Tutorial Shell

Status: implemented.

Turn the Streamlit frontend into a guided lab instead of a bare dashboard.

Core pattern:

```text
Concept -> Math -> Implementation -> Experiment
```

Tasks:

- add a project intro page
- add a domain model tutorial page
- add a reusable source-code excerpt helper
- show curated snippets from the real engine implementation
- explain why instruments, market state, configs, and results are separate objects
- keep tutorial content concise and close to the implementation it describes
- keep the Streamlit app outside the quant engine package

Acceptance criteria:

- Streamlit app has tutorial navigation
- domain tutorial renders without pricing logic
- source snippets are read from actual source files
- `src/quant_lab` remains UI-agnostic
- tests, linting, and Pyright pass

## Milestone 3: Black-Scholes Closed Form

Status: implemented.

Implement analytical European call and put pricing.

Math:

```text
dS_t = r S_t dt + sigma S_t dW_t

d1 = [ln(S0 / K) + (r - q + 0.5 sigma^2)T] / [sigma sqrt(T)]
d2 = d1 - sigma sqrt(T)
```

For calls:

```text
C = S0 exp(-qT) N(d1) - K exp(-rT) N(d2)
```

For puts:

```text
P = K exp(-rT) N(-d2) - S0 exp(-qT) N(-d1)
```

Tasks:

- implement `black_scholes_price`
- support dividend yield `q`
- add put-call parity test
- add known-value regression tests
- handle near-expiry behavior clearly

Acceptance criteria:

- call and put prices match trusted benchmark values
- put-call parity holds within tolerance
- implementation is short and readable

## Milestone 4: GBM Simulation

Status: implemented.

Implement geometric Brownian motion paths under the risk-neutral measure.

Math:

```text
S_{t+dt} = S_t exp((r - q - 0.5 sigma^2) dt + sigma sqrt(dt) Z)
```

Tasks:

- implement Brownian increment generation
- implement GBM path simulation
- support seed for reproducibility
- optionally support antithetic variates
- return arrays shaped consistently

Acceptance criteria:

- simulations are reproducible with fixed seed
- terminal distribution sanity checks pass
- path plotting can consume the result directly

## Milestone 5: Monte Carlo Pricing

Status: implemented.

Implement Monte Carlo pricing for European options.

Math:

```text
V0 = exp(-rT) E_Q[payoff(S_T)]
```

Tasks:

- implement payoff functions
- implement `monte_carlo_price`
- calculate standard error
- calculate 95 percent confidence interval
- optionally return sample paths for visualization
- compare estimates against Black-Scholes

Acceptance criteria:

- Monte Carlo estimate converges toward closed-form price as paths increase
- confidence interval is reported
- tests verify broad agreement with analytical price under fixed seed and tolerances

## Milestone 6: Streamlit Black-Scholes Lab

Status: implemented.

Build the first interactive UI page.

Controls:

- option type
- spot
- strike
- maturity
- volatility
- risk-free rate
- dividend yield
- number of paths
- number of time steps
- seed
- antithetic variates

Visuals:

- simulated paths
- terminal price histogram
- payoff diagram
- Monte Carlo convergence chart
- analytical versus simulated price summary
- price surface over spot and volatility
- compact math section with formulas

Acceptance criteria:

- app runs locally
- controls update charts
- engine remains Streamlit-free
- charts are readable on a normal laptop viewport

## Milestone 7: Greeks

Status: implemented.

Add analytical Greeks and finite-difference Greek approximations for Black-Scholes.

Tasks:

- delta
- gamma
- vega
- theta
- rho
- finite-difference approximation helpers that bump inputs and estimate sensitivities
- charts for Greek curves and surfaces

Acceptance criteria:

- analytical Greeks match finite-difference approximations within tolerance
- Streamlit page can visualize Greek behavior

## Milestone 8: Finite-Difference PDE Solver

Status: implemented.

Add a finite-difference PDE-based option pricing solver after the Black-Scholes and Monte Carlo base is stable.

This is different from Milestone 7 finite-difference Greeks: here we discretize the Black-Scholes PDE over stock and time grids to compute option prices.

Methods:

- explicit scheme
- implicit scheme
- Crank-Nicolson

Visuals:

- option value surface over stock price and time
- final price curve
- error against analytical Black-Scholes
- stability behavior

Acceptance criteria:

- solver matches analytical Black-Scholes within expected discretization error
- boundary conditions are explicit and documented
- numerical stability limitations are visible in the UI

## Milestone 9: Lab Book Learning Layer

Turn the working labs into a more complete learning path. The goal is to keep
the mathematical content sufficiently formal for an engineer, while still
introducing concepts with intuition, examples, and guidance for interpreting
the charts.

Design principle:

```text
Definition -> Intuition -> Math -> Implementation -> What to look for
```

The lab book should not replace the interactive experiments. It should prepare
the reader to understand what the experiment is showing, then point back to the
actual typed Python implementation.

### Milestone 9.1: Foundations and Option Vocabulary

Status: implemented.

Introduce the basic objects used throughout the lab.

Topics:

- what call and put options are
- strike, maturity, spot, payoff, premium, moneyness, intrinsic value, time value
- payoff versus price
- why an option can be worth more than immediate exercise value
- risk-neutral pricing as an introductory idea, without full derivation yet

Visual explanations:

- call payoff shape
- put payoff shape
- intrinsic value versus time value
- how moneyness changes as spot moves around strike

Acceptance criteria:

- symbols are introduced before equations use them
- call and put payoff diagrams are explained in plain language
- the Black-Scholes and Monte Carlo pages can link back to this vocabulary

### Milestone 9.2: Random Variables and Processes Primer

Status: implemented.

Add a concise mathematical primer for the probability tools used later.

Topics:

- random variables and distributions
- expectation, variance, standard deviation, standard error
- random processes as time-indexed random variables
- Brownian motion intuition
- independent increments and normally distributed increments
- drift and volatility

Example equations:

```text
W_{t+\Delta t} - W_t ~ N(0, \Delta t)
```

Acceptance criteria:

- the explanation is introductory and does not assume prior stochastic calculus
- equations are accompanied by practical interpretation
- GBM, Monte Carlo, and Black-Scholes pages can refer back to this primer

### Milestone 9.3: Stochastic Calculus Primer

Status: implemented.

Add a dedicated stochastic calculus page before the deeper GBM and Black-Scholes
derivations.

This page should be more formal than the probability primer, but still
introductory and visual. Its job is to make stochastic integrals, Ito processes,
quadratic variation, and Ito's lemma feel usable before they appear in pricing
derivations.

Topics:

- why ordinary calculus intuition breaks for Brownian paths
- finite variation versus quadratic variation
- stochastic integrals as limits of sums:

```text
\int_0^T H_t\,dW_t
```

- integrands as adaptive processes: `H_t` can depend on information known by
  time `t`
- martingale intuition for stochastic integrals with respect to Brownian motion
- Ito process notation:

```text
dX_t = a_t dt + b_t dW_t
```

- finite-step accumulation argument for why order `dt` terms survive
- multiplication rules derived from accumulated scale:

```text
dt^2 = 0
dt\,dW_t = 0
dW_t^2 = dt
```

- Ito's lemma in one dimension
- how Ito's lemma differs from the ordinary chain rule
- the Ito correction as curvature exposure to Brownian variance
- using Ito's lemma in reverse to evaluate stochastic integrals

Visual explanations:

- Brownian path roughness versus a smooth log path
- quadratic variation convergence
- simple stochastic integral accumulation
- comparison between ordinary chain rule and Ito correction

Acceptance criteria:

- stochastic integral notation is introduced before Ito's lemma
- `dt^2 = 0`, `dt dW_t = 0`, and `dW_t^2 = dt` are explained through
  accumulated scale, not hand-waved
- Ito's lemma is stated clearly with symbols defined locally
- the page prepares the reader for full GBM and Black-Scholes derivations

### Milestone 9.4: Geometric Brownian Motion Derivation and Interpretation

Status: implemented.

Expand the GBM lab into a derivation-driven page.

Topics:

- why additive price models are problematic for equities
- proportional returns
- log returns
- deriving the GBM SDE:

```text
dS_t = \mu S_t dt + \sigma S_t dW_t
```

- deriving the simulation step:

```text
S_{t+\Delta t} =
S_t \exp((\mu - 0.5\sigma^2)\Delta t + \sigma\sqrt{\Delta t}Z)
```

- why GBM is used as a baseline model
- what GBM gets wrong in real markets

Graph explanations:

- sample paths
- terminal price distribution
- drift effects
- volatility effects
- why path-level randomness and distribution-level stability coexist

Acceptance criteria:

- the GBM page explains both the SDE and the discrete simulator
- the log-price solution is derived using Ito's lemma once Milestone 9.3 exists
- chart captions explain what should change when parameters move
- limitations are stated without derailing the introductory flow

### Milestone 9.5: Probability Measures and Risk-Neutral Pricing

Add a dedicated learning section for real-world versus pricing probabilities.

This should come before the full Black-Scholes derivation, because the
Black-Scholes formula and Monte Carlo pricing both rely on expectations under
the risk-neutral measure.

Topics:

- probability measures as probability assignments over possible future paths
- physical / real-world measure `P`
- risk-neutral / pricing measure `Q`
- expectation notation:

```text
E^P[X], E^Q[X]
```

- why `Q` is not just notation
- discounted tradable total-gain processes as martingales under `Q`
- non-dividend stock drift under `Q`
- continuous dividend yield and why stock price drift becomes `r - q`
- same volatility, different Brownian motion:

```text
dS_t = \mu S_t dt + \sigma S_t dW_t^P
dS_t = (r-q) S_t dt + \sigma S_t dW_t^Q
```

- market price of risk intuition
- introductory change-of-measure idea
- Girsanov's theorem as the formal result, without requiring full proof

Acceptance criteria:

- `P`, `Q`, and `E^Q` are defined before being used in pricing pages
- the difference between forecasting and pricing is explicit
- `r - q` is derived from risk-neutral total return, not asserted
- the page prepares the reader for Black-Scholes and Monte Carlo pricing

### Milestone 9.6: Black-Scholes Derivation and Closed-Form Intuition

Status: implemented, with probability-measure foundations to be separated into Milestone 9.5.

Expand the Black-Scholes lab with a guided derivation.

Topics:

- modelling assumptions
- Ito's lemma applied to option value `V(S,t)`
- delta hedging and why the random term can be removed
- no-arbitrage argument for the Black-Scholes PDE
- risk-neutral dynamics
- risk-neutral expected payoff derivation of European call and put formulas
- intuition for `d1` and `d2`
- put-call parity as a consistency check

Graph explanations:

- payoff diagram
- price curve
- price surface over spot and volatility
- why calls and puts respond differently to spot
- why both option types become more valuable as volatility rises

Acceptance criteria:

- the derivation is stepwise and symbol definitions are local to the section
- Ito's lemma is used only after the stochastic calculus primer has introduced it
- the closed-form formula is connected back to the implementation
- the graphs explain expected shapes and parameter sensitivities

### Milestone 9.7: Monte Carlo Pricing Intuition and Convergence

Status: implemented.

Expand the Monte Carlo lab with the probability argument behind pricing by
simulation.

Topics:

- discounted expected payoff
- risk-neutral expectation:

```text
V_0 = e^{-rT} E^Q[\Phi(S_T)]
```

- law of large numbers
- why standard error shrinks like `1 / sqrt(N)`
- confidence intervals
- why Monte Carlo is flexible for complex payoffs
- why Monte Carlo can be slow for precise vanilla prices

Graph explanations:

- path fan chart
- terminal price distribution
- convergence to Black-Scholes
- absolute error / delta from closed form
- confidence interval and standard error

Acceptance criteria:

- the page explains why averaging simulated payoffs gives a price
- convergence charts explicitly discuss noise and sample size
- the strengths and weaknesses of Monte Carlo are stated before PDE comparison

### Milestone 9.8: Greeks as Practical Risk Measures

Status: implemented.

Expand the Greeks page beyond calculation formulas into practical modelling and
risk intuition.

Topics:

- Greeks as local sensitivities
- delta as directional exposure and hedge ratio
- gamma as delta instability
- vega as volatility exposure
- theta as time decay
- rho as rate exposure
- analytical Greeks versus finite-difference estimates
- how Greeks are used for hedging, scenario analysis, and portfolio risk
- limits of local sensitivities

Graph explanations:

- Greek curves over spot
- Greek surfaces over spot and volatility
- why gamma and vega concentrate near the strike
- why call and put deltas differ while gamma and vega can match

Acceptance criteria:

- each Greek has both a formula-level and practice-level explanation
- chart explanations discuss expected call/put behavior
- the page clarifies that Greeks are local approximations, not full risk models

### Milestone 9.9: PDE Solver Intuition, Schemes, and Error Sources

Status: implemented.

Expand the PDE lab with numerical-method intuition.

Topics:

- why pricing PDEs appear from Black-Scholes
- why finite differences solve on a grid over spot and time
- terminal payoff as the starting condition
- boundary conditions
- explicit scheme
- implicit scheme
- Crank-Nicolson scheme
- stability versus accuracy tradeoffs
- where discretization error comes from
- why the error is concentrated near the payoff kink
- when PDE methods are preferred over Monte Carlo
- when Monte Carlo is preferred over PDE methods

Graph explanations:

- option value surface
- time value surface
- final price curve
- error curve
- stability ratio

Acceptance criteria:

- explicit, implicit, and Crank-Nicolson schemes are explained at a high level
- the error chart explanation mentions grid resolution, payoff kink, and boundaries
- the PDE versus Monte Carlo comparison is practical and not purely theoretical

## Milestone 10: Applied Option Risk Lab

Status: implemented.

Apply the previous sections to a small, realistic workflow. The first applied
lab should stay self-contained and synthetic, so it can use the existing pricing,
simulation, Greeks, and PDE tools before we add live market data.

Initial direction:

- build an option position or small option portfolio from calls, puts, and cash
- price the position with Black-Scholes, Monte Carlo, and/or PDE methods where
  appropriate
- aggregate portfolio value and Greeks across positions
- run spot, volatility, and time-to-expiry scenarios
- show P&L curves and surfaces for the whole position
- connect Greeks to practical hedging intuition
- compare local Greek approximations with full repricing under larger moves

Possible labs:

- covered call
- protective put
- straddle / strangle
- vertical spread
- delta-hedged option position
- simple portfolio stress test

Learning goals:

- understand how individual option prices combine into a portfolio
- see why Greeks are useful as local risk summaries
- see where local approximations break under large moves
- understand the difference between pricing one contract and managing a
  position over scenarios
- prepare for later market-data experiments without needing external data yet

Acceptance criteria:

- portfolio positions are represented by typed, UI-agnostic engine objects
- portfolio price and Greek aggregation are tested
- Streamlit includes at least one applied strategy page
- charts explain payoff, value, P&L, and Greek exposure at the portfolio level
- the lab reuses existing pricing/risk modules instead of duplicating formulas

## Milestone 11: Formal Katas Track

Build a second, deeper pass over the same material covered by the introductory
lab. The current lab remains the introduction. The formal track assumes the
reader has completed it, then rebuilds the underlying mathematics more
carefully and with fewer hand-waves.

The goal is not to become a graduate stochastic-calculus text. The goal is a
middle path: more formal than practitioner treatments, less compressed than
research notes, and still readable for a curious undergraduate with basic
calculus, probability, and programming experience.

Core principle:

```text
The intro lab motivates the need; the formal kata rebuilds the tool.
```

Every formal kata should start from a concrete place where the intro lab used a
concept informally. For example, the Monte Carlo page used
`E^Q[payoff]`; the formal probability and pricing-measure katas should explain
what expectation, probability measures, and pricing measures actually mean.

Outside examples are welcome when they clarify the mathematics. Finance should
provide the spine, but analysis and numerical methods can use engineering
examples such as finite differences, heat flow, stability of time stepping, or
finite-element intuition when those examples make the idea easier to see.

### Milestone 11.0: Formal Track Conventions

Define the page style before adding content.

The Streamlit sidebar should communicate the learning architecture instead of
being a flat list of pages. The current Milestones 1-10 pages should sit under
an introductory supersection such as:

```text
Quant Foundations: Introduction and Intuition
```

The formal katas should sit under a separate supersection such as:

```text
Formal Katas: Rebuilding the Machinery
```

Later data and applied projects can become their own supersections. The goal is
that a reader can tell whether a page is an intuitive introduction, a formal
rebuild, or an applied/data workflow before opening it.

Each formal kata should use this structure where appropriate:

```text
Motivation From The Intro Lab
Question
Construction
Definition
Visual Intuition
Proposition
Proof Sketch
Example
Implementation Check
What We Are Still Admitting
```

Conventions:

- assume the reader has completed Milestones 1-10
- formal katas are long-form course chapters, not short dashboard pages
- define every symbol before using it
- do not use a concept before it has been explained, unless the page explicitly
  marks it as an admission or a forward reference
- label theorem-like statements as definition, proposition, proof sketch, or
  admission
- keep a dependency chain between katas; if a page relies on a previous kata,
  say so near the top
- use inline visual arguments where helpful, not only experiment charts
- use underbraces or diagrams for dense equations
- make admissions explicit instead of hiding them behind "it can be shown"
- keep implementation secondary: code verifies or illustrates the math rather
  than driving the exposition
- avoid full measure theory unless the page genuinely needs it; if mentioned,
  mark it as a later formalization
- introduce notation deliberately and keep it stable across pages
- when notation changes context, for example `t` versus time-to-expiry `tau`,
  state the conversion explicitly
- repeat important definitions locally when the reader needs them, but link the
  motivation back to the earlier kata
- prefer one careful derivation over several compressed formulas
- include non-finance examples when they make the mathematics clearer, while
  keeping the finance motivation visible

Acceptance criteria:

- formal pages have a consistent structure
- the sidebar groups pages into learning tracks rather than one flat list
- every page starts with motivation from the existing intro lab
- every page has a "What We Are Still Admitting" section when relevant
- visual explanations are part of the text, not only the experiment panel
- each formal kata has multiple internal subsections and enough detail to stand
  as a course chapter

Suggested internal subsection pattern:

```text
Motivation From The Intro Lab
What We Need To Explain
Prerequisites From Earlier Katas
Definitions and Notation
Construction
Visual Argument
Propositions and Proof Sketches
Worked Examples
Implementation Check
Common Pitfalls
What We Are Still Admitting
Where This Will Be Used Next
```

### Milestone 11.1: Analysis and Approximation Kata

Rebuild the calculus tools used throughout the introductory lab.

Motivation from the intro lab:

- Applied Risk used Taylor expansions for local P&L
- Greeks used derivatives as sensitivities
- PDE Solver used finite differences
- Black-Scholes used local derivatives in Ito's lemma and the PDE

Topics:

- functions, limits, continuity, and local behavior
- derivatives as best local linear approximations
- Taylor expansion and error terms
- first-order versus second-order approximations
- convexity and curvature
- finite differences as numerical derivatives
- why local approximations break under large moves

Visuals:

- tangent line versus curve
- Taylor approximation error
- curvature and convexity diagrams
- first- and second-difference stencils
- local Greek approximation versus full repricing curve

Acceptance criteria:

- Taylor expansion used in Applied Risk is derived from local approximation
- finite-difference stencils are motivated before PDE numerics
- error terms are explained without excessive rigor

### Milestone 11.2: Probability Foundations Kata

Rebuild probability and expectation from the ground up.

Motivation from the intro lab:

- Monte Carlo averaged simulated payoffs
- Black-Scholes and Monte Carlo used expected payoff notation
- Probability Primer introduced random variables but not the deeper structure

Topics:

- outcomes, events, and probability assignments
- random variables as functions from outcomes to values
- distributions as the induced behavior of random variables
- expectation as weighted average / integral
- variance, covariance, and correlation
- conditional expectation as an informed average
- law of large numbers
- central limit theorem as an admitted theorem with simulation evidence
- standard error and why it scales like `1 / sqrt(N)`

Visuals:

- outcome space mapped into the real line
- random variable inducing a distribution
- sample average convergence
- standard error shrinkage
- histogram stabilization

Acceptance criteria:

- Monte Carlo averaging is connected to expectation and the law of large numbers
- standard error formula is motivated clearly
- conditional expectation is introduced only as far as later pricing needs it

### Milestone 11.3: Stochastic Processes and Brownian Motion Kata

Build the stochastic process used by GBM and Ito calculus.

Motivation from the intro lab:

- GBM used Brownian shocks
- Stochastic Calculus used `dW_t^2 = dt`
- Monte Carlo simulated paths, not just one random variable

Topics:

- stochastic processes as time-indexed random variables
- independent increments
- Gaussian increments
- Brownian motion properties
- why increments satisfy `W_{t+dt} - W_t ~ N(0, dt)`
- scaling and typical move size `sqrt(dt)`
- path continuity and nondifferentiability
- quadratic variation
- smooth path versus Brownian path

Visuals:

- random variable versus stochastic process
- increment boxes over time
- Brownian zoom roughness
- quadratic variation convergence
- smooth path compared with Brownian path

Admissions:

- existence of Brownian motion can be stated rather than fully constructed at
  first

Acceptance criteria:

- Brownian scaling is clear before Ito calculus uses it
- quadratic variation is motivated through accumulated squared increments
- path roughness is shown visually

### Milestone 11.4: Stochastic Calculus Kata

Rebuild Ito calculus from the Brownian scaling rules.

Motivation from the intro lab:

- GBM used Ito's lemma to derive log-price dynamics
- Black-Scholes used Ito's lemma on `V(t, S_t)`
- the introductory stochastic calculus page stated the rules but did not fully
  develop the construction

Topics:

- stochastic integrals as limits of adapted left-point sums
- adapted processes and why future information is forbidden
- Ito processes
- multiplication table from accumulated scale
- why `dt^2 = 0`, `dt dW_t = 0`, and `dW_t^2 = dt`
- Ito's lemma from Taylor expansion
- the Ito correction as curvature exposure to randomness
- using Ito's lemma in reverse to evaluate stochastic integrals

Visuals:

- left-point stochastic sums
- accumulated scale of `dt^2`, `dt dW_t`, and `dW_t^2`
- ordinary chain rule versus Ito correction
- curvature plus noise diagram

Admissions:

- rigorous construction of the Ito integral can be sketched, not fully proved

Acceptance criteria:

- Ito's lemma is derived from the previous Brownian and analysis katas
- the new second-derivative term is unavoidable, not magic
- the page prepares directly for GBM and Black-Scholes rebuilt pages

### Milestone 11.5: Asset Modelling and GBM Kata

Rebuild the stock-price model more formally.

Motivation from the intro lab:

- GBM Simulation used the exponential update
- Black-Scholes assumed the same model
- Monte Carlo simulated terminal prices from this process

Topics:

- additive versus multiplicative price models
- simple returns versus log returns
- proportional drift and proportional noise
- GBM SDE:

```text
dS_t = mu S_t dt + sigma S_t dW_t
```

- applying Ito's lemma to `log(S_t)`
- exact solution
- terminal lognormal distribution
- mean, median, and variance of terminal price
- what GBM assumes and what it misses

Visuals:

- additive model crossing zero
- compounding paths
- log transform
- terminal normal log-price versus lognormal price
- mean versus median under lognormality

Acceptance criteria:

- the exponential simulator is derived rather than asserted
- positivity and lognormality are proved at the introductory formal level
- model limitations are explicit

### Milestone 11.6: Pricing Measures and No-Arbitrage Kata

Explain why pricing uses `Q` instead of the physical drift.

Motivation from the intro lab:

- GBM replaced `mu` with `r - q` for pricing
- Black-Scholes and Monte Carlo used `E^Q`
- Foundations introduced risk-neutral pricing only as a preview

Topics:

- physical measure `P` versus pricing measure `Q`
- pricing is not forecasting
- discounting deterministic cashflows
- one-period no-arbitrage model
- risk-neutral probabilities in a binomial tree
- discounted tradable total-gain processes as martingales under `Q`
- continuous dividend yield and why price drift becomes `r - q`
- introductory change-of-measure intuition
- Girsanov's theorem as the named continuous-time result

Visuals:

- one-period tree with physical probabilities
- one-period tree with risk-neutral probabilities
- discounted expected value diagram
- drift swap from `P` to `Q`
- stock plus dividend total-return sketch

Admissions:

- full measure-change machinery and Girsanov proof can be admitted initially

Acceptance criteria:

- `P`, `Q`, `E^Q`, and `r - q` are motivated before being reused
- risk-neutral probability is presented as a pricing tool, not a belief about
  the real world
- no-arbitrage is shown first in a discrete model

### Milestone 11.7: Black-Scholes Rebuilt Kata

Rebuild the Black-Scholes PDE and closed form with stricter bookkeeping.

Motivation from the intro lab:

- Black-Scholes page derived the PDE and formula, but compressed some modelling
  and self-financing details
- PDE Solver and Greeks rely on the same PDE and derivatives

Topics:

- modelling assumptions
- option value as `V(t, S_t)`
- applying Ito's lemma step by step
- self-financing portfolio
- dividend cashflows
- delta hedge and Brownian shock cancellation
- locally riskless portfolio and no-arbitrage return
- Black-Scholes PDE
- risk-neutral expected payoff representation
- closed-form call derivation
- put formula
- put-call parity
- interpretation of `d1` and `d2`

Visuals:

- hedge portfolio diagram
- Brownian `dW_t` cancellation
- PDE term annotations
- exercise region `S_T > K`
- standard-normal threshold picture for `d2`
- stock-weighted expectation intuition for `d1`

Acceptance criteria:

- all symbols are local and defined before use
- the PDE derivation tracks dividend cashflows and self-financing assumptions
- the closed form is connected to the risk-neutral terminal distribution

### Milestone 11.8: Numerical Pricing Methods Kata

Rebuild Monte Carlo and PDE numerics with more numerical-analysis structure.

Motivation from the intro lab:

- Monte Carlo showed convergence and standard error
- PDE Solver showed grid error, stability, and kink effects
- Applied Risk used full repricing versus local approximations

Topics:

- approximation error
- bias versus variance
- Monte Carlo estimator
- standard error derivation
- confidence intervals
- finite-difference grids
- consistency, stability, and convergence
- explicit, implicit, and Crank-Nicolson schemes
- boundary conditions
- payoff kink and error concentration

Visuals:

- Monte Carlo convergence cloud
- confidence interval shrinking
- finite-difference grid and stencils
- explicit stability failure
- boundary truncation picture
- error concentrated around the strike

Admissions:

- Lax equivalence can be named but not proved initially

Acceptance criteria:

- Monte Carlo error and PDE discretization error are clearly different
- stability is shown visually before formal conditions
- the PDE grid implementation is connected back to numerical analysis concepts

### Milestone 11.9: Portfolio Risk Rebuilt Kata

Formalize the applied risk workflow.

Motivation from the intro lab:

- Applied Risk aggregated option positions and Greeks
- Greeks page described local sensitivities
- Black-Scholes and Monte Carlo priced individual contracts

Topics:

- portfolio as a weighted sum of instruments
- linearity of value
- linearity of Greeks
- multi-input Taylor approximation
- local versus global risk
- delta hedging
- gamma risk and hedge drift
- vega exposure
- scenario repricing
- where Greek approximations break

Visuals:

- payoff composition from individual legs
- local tangent versus full reprice curve
- delta hedge flattening first-order exposure
- gamma reintroducing curvature
- scenario surface over spot and volatility

Acceptance criteria:

- Greek aggregation is derived from linearity
- local risk estimates are compared against full repricing
- portfolio optimization is explicitly deferred to a later applied track

## Milestone 12: Data Layer Foundation

Prepare for market data without coupling to a provider.

Tasks:

- define `MarketDataProvider` protocol
- define normalized quote and option-chain objects
- add local CSV provider first
- add repository abstraction only when persistence is needed

Potential providers:

- local CSV
- Yahoo Finance
- Polygon
- Alpaca
- database-backed provider

Acceptance criteria:

- pricing experiments can run against provider output
- provider implementations are swappable
- no model code imports provider-specific APIs

## Milestone 13: Scheduled Experiments

Add repeatable jobs once data exists.

Possible shapes:

- Django management command
- plain CLI command
- cron calling a Python entry point
- Celery beat if scheduling and retries need to be robust

Tasks:

- fetch market snapshot
- run selected models
- store prices, implied vols, errors, and parameters
- compare model output with observed market prices

Acceptance criteria:

- one command can run a daily model snapshot
- results are persisted in a queryable format
- Streamlit or later Django UI can inspect historical runs

## Engineering Conventions

- keep code typed
- keep functions small and named after the math they implement
- prefer dataclasses for simple configuration/result types
- avoid hidden global state
- use deterministic seeds in tests
- test numerical behavior with tolerances, not exact equality
- keep plotting code separate from pricing code
- keep Streamlit code in `apps/streamlit`

## Open Decisions

- use `uv`, Poetry, or plain `pip` tooling
- choose `mypy` versus `pyright`
- decide whether visualization helpers belong inside `src/quant_lab/visualization` or inside app-specific code
- decide whether to introduce Django only after market data experiments become useful
- decide whether persistence should start with SQLite, Postgres, DuckDB, or Parquet

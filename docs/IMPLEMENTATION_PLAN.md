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

Add analytical and finite-difference Greeks for Black-Scholes.

Tasks:

- delta
- gamma
- vega
- theta
- rho
- finite-difference approximation helpers
- charts for Greek curves and surfaces

Acceptance criteria:

- analytical Greeks match finite-difference approximations within tolerance
- Streamlit page can visualize Greek behavior

## Milestone 8: Finite-Difference PDE Solver

Add a PDE-based solver after the Black-Scholes and Monte Carlo base is stable.

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

## Milestone 9: Data Layer Foundation

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

## Milestone 10: Scheduled Experiments

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

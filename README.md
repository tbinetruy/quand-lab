# Quant Lab

Quant Lab is an educational quantitative finance playground. The project is built around a typed Python engine that can be used from notebooks, scripts, scheduled jobs, Streamlit, or a future Django/React/HTMX application.

The first milestones focus on Black-Scholes, geometric Brownian motion, Monte Carlo pricing, Greeks, finite-difference solvers, and interactive visual exploration.

## Setup

This project uses standard Python packaging and works well with `uv`.

```bash
uv sync --dev
```

## Run Checks

```bash
uv run pytest
uv run ruff check .
uv run pyright
```

## Run The Streamlit App

```bash
uv run streamlit run apps/streamlit/main.py
```

## Project Layout

```text
src/quant_lab/
  domain/         Core financial domain objects
  stochastic/     Random processes and path simulation
  models/         Pricing model definitions
  pricing/        Pricing implementations
  risk/           Greeks, hedging, and sensitivities
  data/           Market data interfaces and providers
  experiments/    Reusable application workflows
  visualization/  Chart builders shared by interfaces

apps/streamlit/   Initial interactive frontend
tests/            Unit and numerical tests
docs/             Product and implementation planning
```

The engine under `src/quant_lab` should remain independent from Streamlit and other UI frameworks.

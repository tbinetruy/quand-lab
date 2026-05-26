# Quant Lab PRD

## Purpose

Quant Lab is an educational quantitative finance playground for learning the math, implementation, and behavior of pricing models through readable typed Python and interactive visuals.

The project should make it easy to move between:

- mathematical assumptions and formulas
- clear reference implementations
- visual experiments with parameters
- later integration into a production-style web app or scheduled data pipeline

The first focus is option pricing, starting with Black-Scholes, geometric Brownian motion, Monte Carlo simulation, Greeks, and finite-difference solvers.

## Target User

The primary user is a technically strong learner with:

- engineering math background
- strong Python and software engineering experience
- prior exposure to Monte Carlo option pricing
- preference for readable implementations over performance-oriented code
- preference for interactive applications over notebooks

## Product Principles

- Keep the quant engine independent from any UI framework.
- Prefer readable, typed Python over heavily optimized numerical code.
- Make assumptions explicit in both math explanations and code.
- Visualize model behavior, convergence, errors, distributions, and surfaces.
- Build in small milestones that are useful on their own.
- Allow Streamlit to be replaced later by Django, React, HTMX, FastAPI, or scheduled jobs without rewriting the engine.

## Initial User Experience

The first usable experience should be a Streamlit dashboard that lets the user explore a Black-Scholes and Monte Carlo option pricing lab.

The dashboard should include:

- controls for option and market parameters
- analytical Black-Scholes price
- Monte Carlo price estimate
- confidence interval and standard error
- simulated GBM paths
- terminal price distribution
- payoff diagram
- convergence chart against the analytical price
- price surface over selected parameters
- short math sections with rendered formulas

## Long-Term User Experience

Later versions should support:

- finite-difference PDE solver exploration
- Greeks and sensitivity surfaces
- delta-hedging simulations
- implied volatility inversion
- volatility smile and skew visualization
- alternative models such as Heston and jump diffusion
- market data ingestion
- model snapshots and historical comparisons
- backtesting and calibration experiments

## Architecture Goals

The project should be structured as a Python package with separate interface layers.

Core engine code must not import Streamlit, Django, Plotly UI components, or any web framework.

The intended layering is:

- domain: instruments, option types, market state, quote types
- stochastic: Brownian motion, GBM, random path generation
- models: Black-Scholes, Heston, jump diffusion, local volatility
- pricing: closed-form, Monte Carlo, finite difference
- risk: Greeks, hedging, sensitivities
- data: provider interfaces, normalized market data, repositories
- experiments: application-level workflows that combine engine pieces
- apps: Streamlit now, other interfaces later

## Future Integration Goals

The engine should be reusable from:

- Streamlit pages
- Django views
- Django management commands
- Celery scheduled jobs
- FastAPI endpoints
- tests and scripts
- local data ingestion jobs

Market data support should be provider-agnostic. External services should be wrapped behind interfaces so the project can switch between local CSV files, Yahoo Finance, Polygon, Alpaca, database-backed providers, or other sources.

## Non-Goals For Early Milestones

- high-frequency trading
- portfolio optimization
- production-grade market data ingestion
- real-time trading integration
- advanced calibration workflows before the basics are visual and tested
- performance optimization beyond reasonable vectorized NumPy usage

## Success Criteria

The first milestone is successful when:

- the package installs locally
- tests validate core pricing behavior
- the Streamlit app runs locally
- the user can interactively change Black-Scholes and Monte Carlo parameters
- visual outputs update correctly
- the engine can be called without importing Streamlit
- the code is readable enough to serve as learning material


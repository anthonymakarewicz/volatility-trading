# Development Guide

This guide is for day-to-day development in this repository.

## Environment Setup

1. Use Python 3.12+.
2. Create and activate a virtual environment.
3. Install development dependencies (includes runtime dependencies).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements-dev.txt
pip install -e .
```

`requirements-dev.txt` includes the runtime dependency set, so installing only
`requirements-dev.txt` is sufficient for local development.

## Core Workflow

Use the `Makefile` targets:

```bash
make lint
make format
make check
make typecheck
make test
make test-unit
make test-integration
make sync-nb
make sync-nb-all
make ci
```

`make ci` mirrors the main local quality gate (lint, format check, typecheck, unit tests).

## Pre-commit

Install hooks once:

```bash
pre-commit install
```

Run all hooks manually when needed:

```bash
pre-commit run --all-files
```

Current hooks include:
- whitespace and EOF normalization
- YAML validation
- large-file guard
- Ruff lint and format for `src/` and `tests/`
- Jupytext sync for paired notebooks in `notebooks/`

## Notebook Workflow

Notebook usage is standardized with Jupytext pairing (`.ipynb` + `.py:percent`).

- Single notebook sync:
  - `make sync-nb NOTEBOOK=notebooks/foo.ipynb`
- Sync all notebooks:
  - `make sync-nb-all`

For full setup and policy (pairing, execution checks, output policy), see
[Jupytext workflow](jupytext.md).

## Dependency Updates

When dependencies change, recompile `requirements.txt` from `requirements.in`
using the minimum supported Python version (3.12):

```bash
pip-compile requirements.in -o requirements.txt
```

Then run:

```bash
make check
make test
```

## Test Scope

By default, `pytest` runs unit tests and excludes integration tests via
`pyproject.toml`.

Run integration tests explicitly:

```bash
pytest -m integration -q
```

## CI Behavior

GitHub Actions workflow:
- runs Ruff lint + format checks on `src/` and `tests/`
- runs Pyright type checks on stable `src/volatility_trading` subpackages
- runs unit tests on PRs and pushes to `main`
- runs integration tests on pushes to `main` and manual workflow runs

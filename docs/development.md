# Development Guide

This guide is for day-to-day development in this repository.

## Environment Setup

1. Use Python 3.12+.
2. Create and activate a virtual environment.
3. Install runtime and dev dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Core Workflow

Use the `Makefile` targets:

```bash
make lint
make format
make check
make test
make test-unit
make test-integration
make ci
```

`make ci` mirrors the main local quality gate (lint, format check, unit tests).

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
- runs unit tests on PRs and pushes to `main`
- runs integration tests on pushes to `main` and manual workflow runs

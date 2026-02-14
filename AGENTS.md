# Agent Instructions

This file defines mandatory, high-level workflow rules for this repository.

## Project Context

- This is a Python 3.12 options/volatility trading research + engineering repo.
- Priorities are correctness, reproducibility, and maintainability.
- Work spans ETL/QC pipelines, forecasting/backtesting code, and research notebooks.
- Tooling baseline: `ruff`, `pyright`, `pytest`, and Jupytext-paired notebooks.

## Core Rules

- Scope discipline: keep diffs minimal and avoid unrelated refactors.
- Correctness first: preserve behavior and interfaces unless change is explicitly requested.
- Reproducibility: keep seeds, horizons/units, and config-driven behavior explicit.
- Quality gates: run targeted checks/tests for changed areas before broader checks.
- Docs/tests coupling: behavior changes should include corresponding test/doc updates.
- Safety: do not run destructive git/file operations unless explicitly requested.

## Trigger-Based Docs Routing

- If editing `notebooks/*.ipynb` or `notebooks/*.py`:
  - Read [Notebook Authoring Guide](docs/contributing/notebook_authoring.md).
  - Read [Jupytext Workflow](docs/contributing/jupytext.md).
  - Run `jupytext --sync notebooks/<name>.ipynb` after edits.
  - Keep paired `.ipynb` and `.py` committed together.

- If editing `src/**` behavior:
  - Read [Development Guide](docs/contributing/development.md).
  - Read [Coding Guide](docs/contributing/coding_guide.md).
  - Read [Docstring Guidelines](docs/contributing/docstrings.md).
  - Add or update tests under `tests/**` when behavior changes.

- If editing tests (`tests/**`):
  - Read [Testing Guide](docs/contributing/testing_guide.md).
  - Read [Test Authoring Guide](docs/contributing/test_authoring.md).
  - Read [Tests README](tests/README.md).
  - Prefer shared fixtures in `tests/conftest.py` when setup repeats.
  - Cover edge cases (empty/null/boundary/invalid inputs).

- If editing dependencies (`requirements*.in`, `requirements*.txt`):
  - Follow dependency workflow in [Development Guide](docs/contributing/development.md).
  - Compile lock files with Python `3.12` to match CI.

- If editing CI/tooling (`.github/workflows/*.yml`, `Makefile`, `pyproject.toml`, `.pre-commit-config.yaml`):
  - Ensure local commands and CI commands stay consistent.

- If changing notebook/report names or publishing mapping:
  - Update [Notebook Catalog](docs/research/notebooks.md).

- If changing package/module layout:
  - Update [Package Structure](docs/reference/package_structure.md).

## Prompt Reuse

- For a new chat notebook-writing bootstrap prompt, use the template in:
  - [Notebook Authoring Guide](docs/contributing/notebook_authoring.md) (`Prompt Template for New Chat`).

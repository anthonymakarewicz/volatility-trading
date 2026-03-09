# Development Guide

This guide is for day-to-day development in this repository.
For common environment/tooling failures, see [Troubleshooting](../troubleshooting.md).

## Environment Setup

1. Use Python 3.12+.
2. Install `uv` (for example via Homebrew, `pipx`, or the official installer).
3. Create and activate a virtual environment.
4. Install dependencies.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Primary setup for contributors:
- `uv pip install -e ".[dev]"`: editable install + runtime + dev tools.
- `uv pip install -e ".[dev,release]"`: contributor + release maintainer setup
  (includes packaging tools `build` and `twine`).

Secondary install modes:
- `uv pip install .`: runtime-only install (useful for users running package code without dev tooling).
- `uv pip install -e .`: editable runtime-only install (useful for local source edits without dev tooling).

This repo uses `uv` as the standard environment/install tool. `pip` remains a
fallback if you do not want to install `uv`:

- The repo currently uses `uv`'s `pip`-compatible interface rather than
  `uv sync`; there is no committed `uv.lock` yet.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

If you add or change console scripts in `pyproject.toml` (`project.scripts`),
rerun editable install so new commands are created in `.venv/bin`:

```bash
uv pip install -e .
rehash  # zsh: refresh command lookup
```

## Core Workflow

Use the `Makefile` targets:

```bash
make venv
make sync
make sync-release
make lint
make format
make check
make typecheck
make test
make test-unit
make test-integration
make package-check
make sync-nb
make sync-nb-all
make ci
```

`make ci` mirrors the main local quality gate (lint, format check, typecheck, unit tests).
`make package-check` mirrors packaging validation (`python -m build` + `twine check`).

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
- Ruff lint and format for `src/`, `tests/`, and notebook helper modules (`notebooks/**/*.py`, excluding `notebooks/**/notebook.py`)
- Jupytext sync for paired notebook scripts (`notebooks/**/notebook.py`)

## Commit Messages

Use Conventional Commits:

`<type>(scope): <summary>`

Types: `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`, `ci`, `chore`.

Preferred scopes: `orats`, `etl`, `qc`, `cli`, `notebooks`, `docs`, `ci`.

Examples:
- `feat(orats): add qc summary report generation`
- `fix(qc): handle empty bucket without crash`
- `docs(notebooks): clarify jupytext sync workflow`

Notes:
- Use imperative mood (`add`, `fix`, `remove`).
- Keep the summary <= 72 characters.
- If pre-commit modifies files during `git commit`, run `git add -A` and commit again.

## Releases

For versioning and publish steps, see
[Release Process](release_process.md).

## Notebook Workflow

Notebook usage is standardized with Jupytext pairing (`.ipynb` + `.py:percent`).

- Optional but recommended: register the project venv as a Jupyter kernel:

```bash
python -m ipykernel install --user --name volatility_trading --display-name "Python (.venv) volatility_trading"
```

This makes the venv available in Jupyter kernel selection with a stable name.

- Single notebook sync:
  - `make sync-nb NOTEBOOK=notebooks/foo.ipynb`
- Sync all notebooks:
  - `make sync-nb-all`

For full setup and policy (pairing, execution checks, output policy), see
[Jupytext workflow](jupytext.md).

## Dependency Updates

When dependencies change:

1. Update dependency definitions in `pyproject.toml` (`project.dependencies` or `project.optional-dependencies.dev`).
   For packaging/publish tools, update `project.optional-dependencies.release`.
2. Reinstall the environment:

```bash
make sync
```

3. Run validation:

```bash
make check
make test
```

## Testing

By default, `pytest` runs unit tests and excludes integration tests.
Run integration tests explicitly with:

```bash
pytest -m integration -q
```

For full testing layout/conventions, see [Testing Guide](testing_guide.md).
For quick local test commands, see [Tests README](../../tests/README.md).

## CI Behavior

GitHub Actions workflow:
- runs a dedicated `quality` job for Ruff lint + format checks on `src/`,
  `tests/`, and notebook helper modules (`notebooks/**/*.py`, excluding
  `notebooks/**/notebook.py`)
- the same `quality` job also runs Pyright type checks on stable
  `src/volatility_trading` subpackages plus notebook helper modules
  (`notebooks/**/*.py`, excluding `notebooks/**/notebook.py`)
- runs a dedicated `unit` job with unit tests on PRs and pushes to `main`
- runs integration tests on PRs, pushes to `main`, and manual workflow runs
- runs package build and `twine check` validation on PRs and pushes to `main`

## Troubleshooting

### Wrong Python Version

- Symptom: dependency resolution mismatch vs CI, or missing wheels.
- Likely cause: local interpreter is not Python `3.12`.
- Fix:
  - check interpreter with `python --version`
  - recreate `.venv` with Python `3.12`
  - reinstall dependencies (`make sync` or `uv pip install -e ".[dev]"`)

### Venv Kernel Missing in Jupyter

- Symptom: `.venv` Python exists in interpreter selection, but not in Jupyter kernel list.
- Likely cause: kernel spec was not registered for the venv.
- Fix:
  - activate venv (`source .venv/bin/activate`)
  - register kernel:
    - `python -m ipykernel install --user --name volatility_trading --display-name "Python (.venv) volatility_trading"`
  - restart VS Code/Jupyter kernel picker if needed

### CI Fails But Local Passes

- Symptom: local checks pass, CI fails on types/deps/hooks.
- Likely cause:
  - different Python version
  - local checks narrower than CI
- Fix:
  - run `make ci` locally
  - run `make package-check` locally for packaging failures
  - reinstall dependencies in your active venv (`make sync` or `uv pip install -e ".[dev]"`)
  - re-run `pre-commit run --all-files`

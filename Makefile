.PHONY: help lint format check typecheck test test-unit test-integration ci sync-nb sync-nb-all

NOTEBOOK ?= notebooks/qc_eda/notebook.ipynb

help:
	@echo "Targets:"
	@echo "  make lint               Run Ruff lint on src/, tests/, and notebooks helpers"
	@echo "  make format             Format src/, tests/, and notebooks helpers with Ruff"
	@echo "  make check              Lint + format check (no changes)"
	@echo "  make typecheck          Run Pyright on src/ + notebooks helpers"
	@echo "  make test               Run unit tests (default pytest)"
	@echo "  make test-unit          Run unit tests only"
	@echo "  make test-integration   Run integration tests only"
	@echo "  make sync-nb            Sync one paired notebook (set NOTEBOOK=path)"
	@echo "  make sync-nb-all        Sync all notebooks under notebooks/"
	@echo "  make ci                 Run lint + format check + typecheck + unit tests"

lint:
	ruff check src tests notebooks

format:
	ruff format src tests notebooks

check:
	ruff check src tests notebooks
	ruff format --check src tests notebooks

typecheck:
	pyright

test:
	pytest -q

test-unit:
	pytest tests/unit -q

test-integration:
	pytest -m integration

ci:
	ruff check src tests notebooks
	ruff format --check src tests notebooks
	pyright
	pytest -q

sync-nb:
	jupytext --sync $(NOTEBOOK)

sync-nb-all:
	find notebooks -name '*.ipynb' -print0 | xargs -0 -n1 jupytext --sync

.PHONY: help lint format check typecheck test test-unit test-integration ci sync-nb sync-nb-all

NOTEBOOK ?= notebooks/orats_spy_eda.ipynb

help:
	@echo "Targets:"
	@echo "  make lint               Run Ruff lint on src/ and tests/"
	@echo "  make format             Format src/ and tests/ with Ruff"
	@echo "  make check              Lint + format check (no changes)"
	@echo "  make typecheck          Run Pyright on src/"
	@echo "  make test               Run unit tests (default pytest)"
	@echo "  make test-unit          Run unit tests only"
	@echo "  make test-integration   Run integration tests only"
	@echo "  make sync-nb            Sync one paired notebook (set NOTEBOOK=path)"
	@echo "  make sync-nb-all        Sync all notebooks under notebooks/"
	@echo "  make ci                 Run lint + format check + typecheck + unit tests"

lint:
	ruff check src tests

format:
	ruff format src tests

check:
	ruff check src tests
	ruff format --check src tests

typecheck:
	pyright

test:
	pytest -q

test-unit:
	pytest tests/unit -q

test-integration:
	pytest -m integration

ci:
	ruff check src tests
	ruff format --check src tests
	pyright
	pytest -q

sync-nb:
	jupytext --sync $(NOTEBOOK)

sync-nb-all:
	jupytext --sync notebooks/*.ipynb

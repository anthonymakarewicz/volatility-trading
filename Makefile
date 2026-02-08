.PHONY: help lint format check test test-unit test-integration ci

help:
	@echo "Targets:"
	@echo "  make lint               Run Ruff lint on src/ and tests/"
	@echo "  make format             Format src/ and tests/ with Ruff"
	@echo "  make check              Lint + format check (no changes)"
	@echo "  make test               Run unit tests (default pytest)"
	@echo "  make test-unit          Run unit tests only"
	@echo "  make test-integration   Run integration tests only"
	@echo "  make ci                 Run lint + format check + unit tests"

lint:
	ruff check src tests

format:
	ruff format src tests

check:
	ruff check src tests
	ruff format --check src tests

test:
	pytest -q

test-unit:
	pytest tests/unit -q

test-integration:
	pytest -m integration

ci:
	ruff check src tests
	ruff format --check src tests
	pytest -q

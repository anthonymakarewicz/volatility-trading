# Tests

This file is a quick local entrypoint for test commands.
For standards and full conventions, use:

- [Testing Guide](../docs/contributing/testing_guide.md)
- [Test Authoring Guide](../docs/contributing/test_authoring.md)

## Quick Start

1. Install dev test deps:

```bash
python -m pip install -r requirements-dev.txt
```

2. Run default tests (unit; integration excluded):

```bash
pytest
```

3. Run integration tests explicitly:

```bash
pytest -m integration -q
```

## Common Targets

```bash
make test
make test-unit
make test-integration
make typecheck
```

## Common Subsets

```bash
pytest tests/unit/etl/orats/qc -q
pytest tests/unit/etl/orats/api -q
pytest tests/integration/apps -q
pytest tests/integration/apps/orats -q
pytest tests/integration/apps/fred -q
pytest tests/integration/apps/yfinance -q
pytest -k extract -q
```

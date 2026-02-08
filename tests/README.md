# Tests

This project uses `pytest`.

## Quick Start

1. Install dev test deps:

```bash
python -m pip install -r requirements-dev.txt
```

2. Run unit tests (default; integration is excluded):

```bash
pytest
```

3. Run a specific unit test folder:

```bash
pytest tests/unit -q
```

4. Run only integration tests (marker):

```bash
pytest -m integration -q
```

## Make Targets

If you have `make` available, use:

```bash
make test
make test-unit
make test-integration
```

## Running A Subset

```bash
pytest tests/unit/etl/orats/qc -q
pytest tests/unit/etl/orats/api -q
pytest tests/integration/apps -q
pytest -k extract -q
```

## Notes

- Test configuration lives in `pyproject.toml` under `[tool.pytest.ini_options]`.
- `pythonpath = ["src"]` is set so tests can import `volatility_trading...` without installing the package.
- Most ETL tests are written to be offline and fast (they stub IO/network with `monkeypatch` and use `tmp_path`).
- Integration/smoke tests live under `tests/integration/`.
- `pytest` excludes integration tests by default via `addopts = "-m 'not integration'"`.
- CI runs Ruff + unit tests on every PR/push; integration tests run on pushes to `main`.

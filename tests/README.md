# Tests

This project uses `pytest`.

## Quick Start

1. Install dev test deps:

```bash
python -m pip install -r requirements-dev.txt
```

2. Run the full test suite:

```bash
pytest
```

3. Run only unit tests:

```bash
pytest tests/unit -q
```

4. Run only integration tests (marker):

```bash
pytest -m integration -q
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

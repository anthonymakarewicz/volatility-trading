# Test Structure

This project uses `pytest` for unit and integration tests. Tests live under
`tests/` and follow the same high-level domain layout as `src/volatility_trading/`.

## Layout

```plaintext
tests/
├── integration
│   └── apps
│       ├── conftest.py
│       ├── test_orats_api_download_smoke.py
│       ├── test_orats_api_extract_smoke.py
│       ├── test_orats_ftp_download_smoke.py
│       ├── test_orats_ftp_extract_smoke.py
│       ├── test_orats_build_options_chain_smoke.py
│       ├── test_orats_build_daily_features_smoke.py
│       ├── test_orats_qc_options_chain_smoke.py
│       └── test_orats_qc_daily_features_smoke.py
├── unit
│   ├── cli
│   │   ├── conftest.py
│   │   ├── test_config.py
│   │   └── test_logging.py
│   └── etl
│       └── orats
│           ├── api
│           │   ├── test_api_download.py
│           │   └── test_api_extract.py
│           ├── ftp
│           │   ├── conftest.py
│           │   ├── test_ftp_download.py
│           │   └── test_ftp_extract.py
│           ├── processed
│           │   ├── conftest.py
│           │   ├── test_build_daily_features.py
│           │   └── test_build_options_chain.py
│           └── qc
│               ├── test_common_helpers.py
│               └── test_runners.py
└── README.md
```

## Naming

- Keep test module basenames unique across the entire test suite to avoid
  `import file mismatch` errors (e.g. prefer `test_ftp_download.py` instead of
  another `test_download.py`).
- Use `test_*.py` filenames and `test_*` functions to follow pytest discovery.

## Common Patterns

- Use `monkeypatch` to stub network/IO in ETL tests.
- Use `tmp_path` for temporary directories and files.
- Prefer small, focused unit tests; integration/smoke tests live under
  `tests/integration/`.

## Running Tests

```bash
pytest  # unit tests only (integration excluded by default)
pytest tests/unit/etl/orats/ftp -q
pytest tests/integration/apps -q
pytest -m integration -q
pytest -k extract -q
```

If you have `make` available:

```bash
make test
make test-unit
make test-integration
```

## Configuration

Pytest configuration lives in `pyproject.toml` under
`[tool.pytest.ini_options]`:

- `testpaths = ["tests"]`
- `addopts = "-q -m 'not integration'"`
- `pythonpath = ["src"]`

## CI

GitHub Actions runs Ruff + unit tests on every PR/push, and runs integration
tests on pushes to `main` (or manual workflow runs).

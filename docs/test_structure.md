# Test Structure

This project uses `pytest` for unit tests. Tests live under `tests/` and follow
the same high-level domain layout as `src/volatility_trading/`.

## Layout

```plaintext
tests/
├── cli
│   ├── conftest.py
│   ├── test_config.py
│   └── test_logging.py
├── etl
│   └── orats
│       ├── api
│       │   ├── test_api_download.py
│       │   └── test_api_extract.py
│       ├── ftp
│       │   ├── conftest.py
│       │   ├── test_ftp_download.py
│       │   └── test_ftp_extract.py
│       ├── processed
│       │   ├── conftest.py
│       │   ├── test_build_daily_features.py
│       │   └── test_build_options_chain.py
│       └── qc
│           ├── test_common_helpers.py
│           └── test_runners.py
└── README.md

8 directories, 14 files
```

## Naming

- Keep test module basenames unique across the entire test suite to avoid
  `import file mismatch` errors (e.g. prefer `test_ftp_download.py` instead of
  another `test_download.py`).
- Use `test_*.py` filenames and `test_*` functions to follow pytest discovery.

## Common Patterns

- Use `monkeypatch` to stub network/IO in ETL tests.
- Use `tmp_path` for temporary directories and files.
- Prefer small, focused unit tests; keep integration tests in a separate folder
  if you add them later.

## Running Tests

```bash
pytest
pytest tests/etl/orats/ftp -q
pytest -k extract -q
```

## Configuration

Pytest configuration lives in `pyproject.toml` under
`[tool.pytest.ini_options]`:

- `testpaths = ["tests"]`
- `pythonpath = ["src"]`

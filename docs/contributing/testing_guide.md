# Testing Guide

This project uses `pytest` for unit and integration tests. Tests live under
`tests/` and follow the same high-level domain layout as `src/volatility_trading/`.

For test-writing standards (fixtures, edge cases, parametrization, determinism),
see [Test Authoring Guide](test_authoring.md).
For quick local commands, see [Tests README](../../tests/README.md).

## Layout

```plaintext
tests/
├── README.md
├── integration
│   └── apps
│       ├── conftest.py
│       ├── fred
│       │   └── test_fred_sync_smoke.py
│       ├── orats
│       │   ├── test_orats_api_download_smoke.py
│       │   ├── test_orats_api_extract_smoke.py
│       │   ├── test_orats_build_daily_features_smoke.py
│       │   ├── test_orats_build_options_chain_smoke.py
│       │   ├── test_orats_ftp_download_smoke.py
│       │   ├── test_orats_ftp_extract_smoke.py
│       │   ├── test_orats_qc_daily_features_smoke.py
│       │   └── test_orats_qc_options_chain_smoke.py
│       └── yfinance
│           └── test_yfinance_sync_smoke.py
└── unit
    ├── backtesting
    │   ├── test_engine_kernel.py
    │   ├── options_engine
    │   │   ├── test_options_engine_adapters.py
    │   │   ├── test_options_engine_entry.py
    │   │   ├── test_options_engine_exit_rules.py
    │   │   ├── test_options_engine_selectors_sizing.py
    │   │   └── test_options_engine_plan_builder.py
    │   ├── test_performance_calculators.py
    │   ├── test_performance_console.py
    │   ├── test_reporting_builders.py
    │   ├── test_reporting_plots.py
    │   └── test_reporting_service_and_writers.py
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
    ├── options
    │   ├── test_binomial_tree_pricer.py
    │   ├── test_option_pricers.py
    │   └── test_risk_modules.py
    └── strategies
        └── test_vrp_harvesting_strategy.py

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
pytest tests/unit/backtesting/options_engine -q
pytest tests/unit/etl/orats/ftp -q
pytest tests/integration/apps -q
pytest tests/integration/apps/orats -q
pytest tests/integration/apps/fred -q
pytest tests/integration/apps/yfinance -q
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

GitHub Actions runs a dedicated `quality` job (Ruff + Pyright) and a dedicated
`unit` job on every PR/push, and runs integration tests on PRs, pushes to
`main`, and manual workflow runs.

## Troubleshooting

### Pytest Import or Discovery Errors

- Symptom: `ModuleNotFoundError` for `volatility_trading...` or import mismatch.
- Likely cause:
  - pytest config not aligned
  - duplicate test module basenames across directories
- Fix:
  - ensure pytest config includes:
    - `testpaths = ["tests"]`
    - `pythonpath = ["src"]`
  - keep test basenames unique across the suite (for example avoid repeated `test_download.py`)
  - run from repo root with active venv:
    - `pytest -q`
    - `pytest -m integration -q`
- Related docs:
  - [Test Authoring Guide](test_authoring.md)
  - [Tests README](../../tests/README.md)

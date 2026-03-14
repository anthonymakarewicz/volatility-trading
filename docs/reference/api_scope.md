# API Scope (0.5.x)

This page defines what is considered **public API** for pre-1.0 releases.

If something is not listed here as public, treat it as internal and subject to change.

## Public Surfaces

### 1) Python package API (import-level)

Public import surfaces are the package entrypoints intended for users:

- Preferred: `volatility_trading.backtesting`
- Advanced/domain-specific: `volatility_trading.backtesting.options_engine`
- `volatility_trading.options`
- `volatility_trading.strategies`
- `volatility_trading.signals`

Compatibility expectation in `0.5.x`:

- APIs may still evolve between minor versions (`0.x` policy),
- but breaking changes should be documented in `CHANGELOG.md`.

### 2) CLI API

Public CLI commands are those declared in `pyproject.toml` under `project.scripts`
(for example ORATS pipeline apps, `optionsdx-prepare-panel`, `fred-sync`, `yfinance-sync`).

### 3) Data contract API

The canonical options-chain contract used by backtesting is public:

- documented in [Options Data Adapters](options_data_adapters.md)
- canonical fields centralized in `volatility_trading.contracts.options_chain`

## Non-Public / Internal by Default

These are not stable API contracts unless explicitly promoted:

- `volatility_trading.apps.*` implementation modules
- internal ETL/QC module internals under `volatility_trading.etl.*`
- internal helpers and private modules (including `_...` modules)
- deeper backtesting internals such as:
  - `volatility_trading.backtesting.options_engine.lifecycle.*`
  - `volatility_trading.backtesting.options_engine.contracts.*`
  - `volatility_trading.backtesting.options_engine.entry`
  - `volatility_trading.backtesting.options_engine.plan_builder`
  - `volatility_trading.backtesting.runner.*`
- `examples/` code (usage demos, not compatibility contract)
- research-stage modules and notebook support code under:
  - `volatility_trading.iv_surface.*`
  - `volatility_trading.rv_forecasting.*`
  - `notebooks/`

## Import Guidance

- Prefer `volatility_trading.backtesting` for common runtime configuration,
  data-bundle setup, data-loading helpers, hedging policy objects, execution
  models, and adapter classes.
- Use `volatility_trading.backtesting.options_engine` when you intentionally
  want the narrower, engine-specific advanced namespace. That namespace is
  curated around specs, exit rules, execution models, adapter contracts, and
  plan-building helpers rather than runtime-internal lifecycle state objects.
- Avoid importing from deeper backtesting submodules unless you are explicitly
  working against internal implementation details.

## Release Policy Tie-In

- This project is pre-1.0 (`0.x.y`).
- Scope and maturity are documented in `README.md` and release notes.
- Any public-surface changes should be called out in `CHANGELOG.md`.

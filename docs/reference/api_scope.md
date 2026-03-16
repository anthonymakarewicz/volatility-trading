# API Scope (0.7.x)

This page defines what is considered **public API** for pre-1.0 releases.

If something is not listed here as public, treat it as internal and subject to change.

## Public Surfaces

### 1) Python package API (import-level)

Public import surfaces are the package entrypoints intended for users:

- Preferred: `volatility_trading.backtesting`
- Public contract namespace: `volatility_trading.contracts`
- Advanced/domain-specific: `volatility_trading.backtesting.options_engine`
- Advanced/domain-specific: `volatility_trading.backtesting.performance`
- Advanced/domain-specific: `volatility_trading.backtesting.reporting`
- `volatility_trading.options`
- `volatility_trading.strategies`
- `volatility_trading.signals`

Compatibility expectation in `0.7.x`:

- APIs may still evolve between minor versions (`0.x` policy),
- but breaking changes should be documented in `CHANGELOG.md`.

### 2) CLI API

Public CLI commands are those declared in `pyproject.toml` under `project.scripts`
(for example ORATS pipeline apps, `optionsdx-prepare-panel`, `fred-sync`, `yfinance-sync`).

### 3) Data contract API

The canonical options-chain contract used by backtesting is public:

- documented in [Options Data Adapters](options_data_adapters.md)
- canonical fields centralized in `volatility_trading.contracts.options_chain`
- convenience re-exports available from `volatility_trading.contracts`

## Non-Public / Internal by Default

These are not stable API contracts unless explicitly promoted:

- `volatility_trading.apps.*` implementation modules
- internal ETL/QC module internals under `volatility_trading.etl.*`
- internal helpers and private modules (including `_...` modules)
- deeper backtesting internals such as:
  - `volatility_trading.backtesting.data_adapters`
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
  data-bundle setup, data-loading helpers, concrete adapter classes, common
  hedging policy objects, common exit rules, concrete execution models, and
  the standard performance-report path. `Backtester.run()` returns the dense
  trading-session MTM used by the standard report path. Advanced rate-model
  types and raw-to-daily MTM helpers are not part of the root facade.
- Use `volatility_trading.backtesting.options_engine` when you intentionally
  want the narrower, engine-specific advanced namespace. That namespace is
  curated around strategy/spec contracts, hedging and exit configuration,
  execution models, and plan-building helpers rather than data-ingestion
  adapters or runtime-internal lifecycle state objects.
- Use `volatility_trading.backtesting.performance` when you intentionally need
  the narrower performance-metrics schemas, tables, or console-formatting
  helpers beyond the standard root-level report path.
- Use `volatility_trading.backtesting.reporting` when you intentionally need
  report-bundle construction, persistence, or reporting plot helpers.
- Do not rely on `volatility_trading.backtesting.data_adapters` as a package
  facade. Common adapter imports live at the root facade, and repo-internal or
  test-only code should import concrete adapter modules directly when working
  below the documented public surface.
- Avoid importing from deeper backtesting submodules unless you are explicitly
  working against internal implementation details.

## Release Policy Tie-In

- This project is pre-1.0 (`0.x.y`).
- Scope and maturity are documented in `README.md` and release notes.
- Any public-surface changes should be called out in `CHANGELOG.md`.

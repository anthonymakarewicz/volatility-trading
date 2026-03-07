# Changelog

All notable changes to this project will be documented in this file.

This project follows a pre-1.0 versioning policy (`0.x.y`):

- `x` (minor): new features, significant refactors, and pre-1.0 breaking changes
- `y` (patch): bug fixes, docs/tests updates, and non-breaking internal changes

## [Unreleased]

### Breaking changes
- Replaced hedge execution config field `execution.hedge.commission_per_unit` with `execution.hedge.fee_bps`.
- Removed `LinearHedgeExecutionModel` and made `FixedBpsExecutionModel` the default hedge execution model.

### Changed
- Hedge execution cost accounting now uses spread/slippage plus fixed-bps notional fees for rebalancing trades.

## [0.1.0] - 2026-03-06

### Added
- Initial public library surface for options volatility research and backtesting.
- Options backtesting engine with typed runtime contracts and single-position lifecycle hooks.
- Functional dynamic delta hedging with pluggable hedge target and execution model contracts.
- Options-chain data adapter boundary with canonical schema validation and provider adapters.
- OptionsDX pipeline and CLI entrypoint, including fast Polars-based panel preparation.
- Documented public API boundaries and pre-1.0 release/process contracts.

### Changed
- Refactored the options engine into clearer packages/modules (contracts, lifecycle internals, orchestration boundaries).
- Made adapter resolution explicit in backtest run config (no implicit runtime assumptions).
- Standardized IV naming contracts to canonical `market_iv` / `model_iv` semantics.
- Split overloaded backtesting config/type modules into narrower concerns for maintainability.
- Migrated dependency source of truth to `pyproject.toml` and added `release` optional deps (`build`, `twine`).
- Restructured contributor/release documentation and moved detailed research outputs out of the root README.

### Fixed
- Enforced JSON serialization boundary for `trade_legs` reporting output to avoid CSV/parquet inconsistency drift.
- Hardened adapter validation/coercion paths with explicit boundary checks for schema/alias mismatches.

### Developer experience
- Hardened PR CI with dedicated package validation (`python -m build`, `twine check`), concurrency controls, and job timeouts.
- Added local `make package-check` command mirroring CI/release packaging validation.
- Updated CI installs to use pyproject extras after requirements-file deprecation.

### Docs
- Formalized branch/PR/release process and failure recovery policy for PR-protected `main`.
- Added AI prompt templates for key contributor docs (development, coding, testing, release).
- Simplified and generalized docstring guidance across modules.

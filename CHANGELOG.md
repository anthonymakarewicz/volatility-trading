# Changelog

All notable changes to this project will be documented in this file.

This project follows a pre-1.0 versioning policy (`0.x.y`):

- `x` (minor): new features, significant refactors, and pre-1.0 breaking changes
- `y` (patch): bug fixes, docs/tests updates, and non-breaking internal changes

## [Unreleased]
### Added
- Added `SkewMispricingSpec` and `make_skew_mispricing_strategy` as a second
  built-in options strategy preset, using a 25-delta risk reversal selected
  through the shared options-engine structure logic.

### Changed
- Standardized project setup and contributor installs on `uv`; `pip` remains
  supported as a fallback.
- Added curated `volatility_trading.backtesting` re-exports for preferred user
  imports, while keeping `volatility_trading.backtesting.options_engine` as the
  advanced namespace.
- Narrowed `volatility_trading.backtesting.options_engine` re-exports so the
  advanced namespace no longer advertises runtime-internal lifecycle helpers.
- Options backtests now honor `Signal.exit` as a shared lifecycle close trigger,
  emitting `Signal Exit` trade rows when a strategy exits via signal mean
  reversion.
- `ZScoreSignal` rolling statistics now use only prior observations, removing
  look-ahead leakage from its z-score computation.
- Reorganized backtesting examples under strategy-specific entrypoints and
  renamed the shared example helper module to
  `examples/core/backtesting_helpers.py`.

## [0.4.0] - 2026-03-09
### Added
- Added `OptionsMarketData` and made it the canonical options input wrapper for
  options chain data plus chain-level metadata (`symbol`,
  `default_contract_multiplier`, optional scoped adapter).

### Changed
- `OptionsBacktestDataBundle` now consumes `options_market` directly and no
  longer mixes raw options-frame and adapter wiring at the bundle top level.
- Adapter resolution is now data-owned:
  `data.options_market.options_adapter` when provided, otherwise built-in ORATS
  adapter default.
- `ExecutionConfig` is now model-owned for runtime fills:
  `option_execution_model` and `hedge_execution_model` are the source of truth
  for option and hedge execution behavior.
- `WWDeltaBandModel` now derives effective fee assumptions from the configured
  hedge execution model when `fee_bps_override` is not set.

### Breaking changes
- Renamed hedge execution concrete classes:
  `MidNoCostExecutionModel` -> `MidNoCostHedgeExecutionModel`
  and `FixedBpsExecutionModel` -> `FixedBpsHedgeExecutionModel`.
- Removed `OptionsBacktestDataBundle(options=...)` constructor support.
- Removed `OptionsBacktestDataBundle(options_adapter=...)` constructor support.
  Use `OptionsBacktestDataBundle(options_market=OptionsMarketData(...))` and
  set the scoped adapter on `OptionsMarketData.options_adapter`.
- Removed `BacktestRunConfig.options_adapter_mode`.
- Removed `BacktestRunConfig.options_adapter`.
- Removed legacy `ExecutionConfig` scalar execution fields:
  `slip_ask`, `slip_bid`, `commission_per_leg`,
  `hedge_slip_ask`, `hedge_slip_bid`, `hedge_fee_bps`.
  Configure execution via model instances instead.
- Removed `ExecutionConfig.lot_size`. Option contract multiplier is now sourced
  from `OptionsMarketData.default_contract_multiplier`.
- Renamed options sizing API fields from `lot_size` to
  `option_contract_multiplier`:
  `SizingRequest.option_contract_multiplier` and
  `estimate_entry_intent_margin_per_contract(..., option_contract_multiplier=...)`.
- Removed `OptionsBacktestDataBundle.fallback_iv_feature_col`.
  Entry volatility now comes from selected leg quote `market_iv` only (otherwise
  `NaN`), with no fallback to features columns.

## [0.3.0] - 2026-03-08

### Added
- Added option-leg execution contracts and models in the options lifecycle:
  `OptionExecutionModel`, `OptionExecutionOrder`, `OptionExecutionResult`,
  `BidAskFeeOptionExecutionModel` (default), and `MidNoCostOptionExecutionModel`.
- Added explicit option attribution fields to runtime outputs:
  `MtmRecord.option_market_pnl`, `MtmRecord.option_trade_cost`,
  `TradeRecord.option_entry_cost`, `TradeRecord.option_exit_cost`,
  `TradeLegRecord.entry_mid_price`, and `TradeLegRecord.exit_mid_price`.
- Added dedicated option execution documentation:
  `docs/reference/backtesting/option_execution.md`.

### Changed
- Entry, standard exit, and forced-liquidation option fills now execute through
  `OptionExecutionModel` with consistent leg-level fill/cost behavior.
- Option accounting now separates market movement from transaction costs:
  entry and exit costs are charged on their actual lifecycle dates.
- `Backtester` keeps a stable high-level surface; advanced option execution
  overrides are injected at plan-build time via
  `build_options_execution_plan(..., option_execution_model=...)`.
- Updated architecture and README documentation to reflect option execution
  model semantics and the current injection boundary.

### Breaking changes
- Option PnL attribution semantics changed: `delta_pnl` now reflects separated
  option market movement plus explicit option trade-cost booking.
- Downstream consumers should account for new/updated option attribution output
  fields in MTM and trade tables.

## [0.2.1] - 2026-03-08

### Added
- Added focused hedging examples (fixed band, WW band, and cost baseline comparison) plus shared `examples/core` helper utilities.
- Added a dedicated options-engine hedging guide for fixed-band and WW-style policies (`docs/reference/backtesting/hedging.md`).

## [0.2.0] - 2026-03-08

### Added
- Added `MidNoCostExecutionModel` as a hedge execution baseline (`fill=mid`, `total_cost=0`) for debugging and benchmarking.
- Added dynamic hedge band models: `FixedDeltaBandModel` and `WWDeltaBandModel`.

### Breaking changes
- Replaced hedge execution config field `execution.hedge.commission_per_unit` with `execution.hedge.fee_bps`.
- Removed `LinearHedgeExecutionModel` and made `FixedBpsExecutionModel` the default hedge execution model.
- Replaced `HedgeTriggerPolicy.delta_band_abs` with `HedgeTriggerPolicy.band_model`.
- Added `DeltaHedgePolicy.rebalance_to` and enforce `WWDeltaBandModel` with `rebalance_to='nearest_boundary'`.

### Changed
- Hedge execution cost accounting now uses spread/slippage plus fixed-bps notional fees for rebalancing trades.
- Delta hedging now supports Whalley-Wilmott-style dynamic no-trade bands with nearest-boundary rebalancing.

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

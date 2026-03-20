# Changelog

All notable changes to this project will be documented in this file.

This project follows a pre-1.0 versioning policy (`0.x.y`):

- `x` (minor): new features, significant refactors, and pre-1.0 breaking changes
- `y` (patch): bug fixes, docs/tests updates, and non-breaking internal changes

## [Unreleased]

### Changed
- Strategy presets now default `min_contracts` to `0` instead of forcing a
  one-lot minimum when risk-based sizing would otherwise round down to zero.
- Realized backtest trade rows now include sizing diagnostics for the raw
  risk-budget and margin-budget contract counts, the binding sizing
  constraint, and whether a minimum-contract floor overrode the raw risk size.
- Advanced sizing configs now support `entry_risk_basis="entry_hedged"` so
  risk-budget sizing can use the stressed option package plus the implied
  inception hedge instead of the unhedged option structure alone.
- Runner strategy preset configs now coerce nested `delta_hedge` YAML mappings
  into typed hedge-policy objects, including fixed and WW band-model configs.
- Options-engine strategies can now attach explicit factor decomposition models.
  The skew mispricing preset now emits risk-reversal factor state/exposure
  columns and factor-level daily P&L attribution (`IV_Level_PnL`,
  `RR_Skew_PnL`) alongside the existing Greek outputs.
- The P&L attribution plot now renders factor-level vol lines such as
  `IV_Level_PnL` and `RR_Skew_PnL` when those columns are present, instead of
  showing only the aggregate `Vega_PnL` line.
- The stress-risk engine now supports opt-in risk-reversal shock grids through
  `MarketShock.d_risk_reversal` and
  `FixedGridScenarioGenerator.risk_reversal_shocks`, allowing worst-loss sizing
  and PM-style margin to respond to skew steepening / flattening without
  changing the default grid behavior.
- Runner workflow configs now support `modeling.scenario_generator` with the
  `fixed_grid` model, including explicit `risk_reversal_shocks` for skew-aware
  stress sizing in config-driven backtests.

## [0.7.0] - 2026-03-16

### Added
- Added a reporting-baseline expansion for saved backtest bundles with
  `margin_diagnostics_daily.csv`, `rolling_metrics.csv`,
  `pnl_attribution_daily.csv`, and optional `benchmark_comparison.json`
  artifacts, plus persisted margin-account, rolling-metrics, and
  P&L-attribution component plots.

### Changed
- Bumped the backtest report bundle schema version from `1.0.0` to `1.1.0`
  for the richer reporting artifact set.
- Unified processed ORATS options loading across the public backtesting helper
  and runner loader, pushing date/DTE filters before wide-to-long reshaping and
  validating processed parquet output through the canonical strict contract.
- Expired positions now settle and close when live contract quotes disappear,
  preventing stale open inventory from persisting past expiry in workflow runs.
- Open positions now force-close on the final trading date when live closing
  quotes are available, emitting an ``End Of Run Liquidation`` exit instead of
  silently carrying inventory past the configured run window.
- Workflow daily MTM and reporting now reindex to the run's trading-session
  dates instead of inserting weekend carry-forward rows.
- `Backtester.run()` and workflow results now expose the dense trading-session
  MTM as the standard `mtm` output instead of the sparse raw lifecycle MTM.

### Breaking changes
- `volatility_trading.backtesting.data_adapters` no longer acts as a curated
  package facade; common adapter imports stay at the root facade, while
  repo-internal callers should import concrete adapter modules directly.
- `volatility_trading.backtesting.options_engine` no longer re-exports the
  low-level `coerce_options_frame_to_pandas(...)` and
  `normalize_options_chain(...)` ingestion helpers.
- `volatility_trading.backtesting.options_engine` no longer re-exports
  adapter classes, adapter base types, adapter errors, or canonical
  options-chain contract constants. Adapter imports live at the root
  `volatility_trading.backtesting` facade, while lower-level ingestion work
  should import concrete adapter modules directly.
- Removed low-level options validation helpers and `ValidationMode` from the
  `volatility_trading.backtesting` and
  `volatility_trading.backtesting.options_engine` re-export surfaces. The
  intended user-facing entrypoints are loader/adaptor paths such as
  `load_orats_options_chain_for_backtest(...)`,
  `canonicalize_options_chain_for_backtest(...)`, and the adapter classes.
- Reduced the `volatility_trading.backtesting` root facade to common user
  entrypoints. Advanced strategy/spec contracts, base adapter types, and margin
  internals are no longer hoisted at the package root. Base execution-model
  protocols, rate-model types, and niche performance helpers are also no longer
  exposed there; import those from
  `volatility_trading.backtesting.options_engine`,
  `volatility_trading.backtesting.performance`,
  `volatility_trading.backtesting.rates`, or concrete internal modules when you
  intentionally need the advanced surface.
- `volatility_trading.backtesting` no longer re-exports `to_daily_mtm(...)`.
  The standard backtester and workflow surfaces now return the dense
  trading-session MTM directly as `mtm`.

### Docs
- `docs/reference/api_scope.md` now explicitly treats
  `volatility_trading.backtesting.options_engine`,
  `volatility_trading.backtesting.performance`, and
  `volatility_trading.backtesting.reporting` as the advanced backtesting public
  import surfaces, while leaving
  `volatility_trading.backtesting.data_adapters` internal-by-default.

## [0.6.0] - 2026-03-14

### Added
- Added backtesting data-loading helpers for canonical options-chain
  normalization, ORATS chain loading, FRED rate series loading, yfinance close
  series loading, and spot-series derivation from canonical options data.
- Added `load_daily_features_frame(...)` as a backtesting-oriented helper for
  processed daily-features panels.
- Added common date-window and DTE filters to the ORATS backtest loader helper.
- Added runner support for ORATS source-level `dte_min` / `dte_max` filters in
  workflow YAML.
- Added runner support for rate-sourced financing via
  `cash_rate_source: data_rates` and `borrow_rate_spread`.
- Added structure-level `StopLossExitRule` and `TakeProfitExitRule` based on
  unrealized net `pnl_per_contract`, plus preset-level convenience fields on
  `VRPHarvestingSpec` and `SkewMispricingSpec`.

### Changed
- `OptionsMarketData` now validates canonical long options input at
  construction time instead of deferring normalization to the runtime plan
  builder.
- Runner workflow configs no longer expose `data.options.adapter_name`; ORATS
  workflows now always canonicalize through the built-in runner loader path.
- Runner `data.rates` sections now require explicit provider-owned fields when
  present instead of treating an empty mapping as an implicit constant `0.0`
  rate source.
- `SameDayReentryPolicy` now supports explicit stop-loss and take-profit
  reentry controls alongside the existing periodic and margin-liquidation
  policies.
- Clarified backtesting architecture, runner workflow docs, and release
  guidance to match the current canonical runtime boundary and pre-`1.0`
  release process.

### Breaking changes
- Removed `OptionsMarketData.options_adapter`. Normalize raw source data
  explicitly with backtesting helpers such as
  `canonicalize_options_chain_for_backtest(...)` before constructing the
  runtime data bundle.

## [0.5.0] - 2026-03-13
### Added
- Added `SkewMispricingSpec` and `make_skew_mispricing_strategy` as a second
  built-in options strategy preset, using a 25-delta risk reversal selected
  through the shared options-engine structure logic.
- Added `backtest-run`, a config-driven workflow CLI for one backtest run with
  YAML config loading, dry-run validation, top-level overrides for ticker,
  run window, and report output, plus typed workflow support for options,
  benchmark, and rates data sources as well as broker margin model/policy
  configuration.
- Added example workflow configs under `config/backtesting/` for
  `vrp_harvesting` and `skew_mispricing`.

### Changed
- Standardized project setup and contributor installs on `uv`; `pip` remains
  supported as a fallback.
- Added curated `volatility_trading.backtesting` re-exports for preferred user
  imports, while keeping `volatility_trading.backtesting.options_engine` as the
  advanced namespace.
- Options backtests now honor `Signal.exit` as a shared lifecycle close trigger,
  emitting `Signal Exit` trade rows when a strategy exits via signal mean
  reversion.
- `LifecycleConfig` now supports signal-driven mode without periodic exits, and
  adds `LifecycleConfig.signal_driven(...)` for signal-only lifecycle with an
  optional max-holding safety cap.
- `ZScoreSignal` rolling statistics now use only prior observations, removing
  look-ahead leakage from its z-score computation.

### Breaking changes
- `volatility_trading.backtesting.options_engine` no longer re-exports
  runtime-internal lifecycle helpers; advanced users should import those names
  from their concrete internal modules instead.
- `backtest-run` requires `strategy.name` explicitly in workflow YAML; the CLI
  does not assume `vrp_harvesting` as a default preset.

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

# Roadmap

This roadmap tracks committed development priorities for the backtesting library.
Exploratory ideas stay in `notes/` until promoted here.

## Now (next 1-2 milestones)

### Strategy diversification
- Complete `skew_mispricing` as a second fully backtestable strategy family (spec, signals, examples, tests).
- Use a second real strategy path to validate that `StrategySpec` and options-engine lifecycle abstractions are truly generic, not just VRP-shaped.

### Backtest run workflows
- Add a declarative backtest runner CLI driven by YAML config (datasets, adapter, strategy preset, execution, broker/modeling, reporting).
- Emit run manifests/report bundles automatically so scripted runs and CI smoke runs share the same execution path.
- Add one stable config schema for reproducible batch runs outside notebooks/examples.
- Add a strategy preset registry/factory so config-driven runs can instantiate strategies by stable name plus validated params.

### Diagnostics and reporting baseline
- Add core risk dashboards: margin usage timeline, call events timeline, and rolling risk-adjusted metrics.
- Add benchmark comparison tables (strategy vs volatility benchmarks) in reporting outputs.
- Add comparative performance tables for each configured benchmark, not only the strategy, with relative-performance metrics recorded alongside strategy results.
- Extend reporting to plot equity curve, drawdown, rolling Sharpe, and rolling annualized volatility for the strategy and all configured benchmarks.
- Add P&L decomposition usign transaction costs form option and hedging instrument.

## Mid-term (priority)

### Richer exits (requested priority)
- Add stop-loss / take-profit exits at structure level.
- Add optional leg-level stop/target exits.
- Add time-stop + volatility mean-reversion exits.
- Add half-life exits.
- Add event exits (earnings and macro/event calendars).

### Margin domain cleanup
- Unify position-level initial margin models (`options/risk/margin.py`) with account-level maintenance/liquidation logic (`backtesting/margin.py`) under clearer shared contracts.
- Separate margin model, maintenance policy, and account/liquidation policy responsibilities more explicitly before portfolio/runtime expansion.

### Position sizing v2
- Add hard Greek-aware sizing constraints (vega/gamma limits).
- Add feature-aware sizing (e.g., VIX buckets, signal z-score buckets).

### Experiment infrastructure
- Add a multi-run experiment layer for parameter grids, strategy comparisons, and execution/margin scenario sweeps.
- Persist experiment-level manifests and aggregate comparison tables alongside single-run report bundles.

### Decision trace / explainability
- Add optional structured debug outputs for entry, sizing, hedging, and exit decisions.
- Record why entries were skipped, why sizing returned zero contracts, which hedge trigger fired, and which exit/liquidation rule closed the position.
- Expose typed trace tables or report artifacts so lifecycle decisions are auditable without stepping through engine internals.

## Later (long-term)

### Portfolio and risk controls
- Move from single-position loop to position-book runtime (multiple concurrent positions).
- Add strategy-level exposure caps first; portfolio-level caps after multi-position core is stable.
- Add drawdown kill-switch and cool-down controls.

### Validation workflows
- Add walk-forward workflow with rolling recalibration/retrain windows.
- Add out-of-sample scorecards by regime and by year.
- Add parameter stability workflows (sensitivity grids/heatmaps).
- Add reality-check workflows with permutation/bootstrap style tests on trade sequencing.
- Add reality-check workflows with spot/IV noise perturbation and recomputed equity-curve stress tests.

### Architecture evolution
- Introduce event-driven runtime compatible with both backtest and paper-trading simulation.
- Keep shared domain contracts stable while evolving execution engines.

### Options domain model upgrades
- Introduce a volatility-surface abstraction so pricing/risk can consume strike- and maturity-aware vol inputs instead of only a scalar `MarketState.volatility`.
- Add liquidity/impact-aware execution models where option and hedge fill costs can scale with trade size, spread, and quote liquidity proxies.

### UX and publishing
- Promote stable plotting/report helpers to documented public API.
- Add richer reporting panels for margin account state, hedge attribution, and regime-aware performance diagnostics.
- Extend benchmark reporting from single-series overlays to multi-benchmark comparison views once the core comparative tables and rolling panels are stable.
- Improve GitHub Pages presentation and publication pipeline.

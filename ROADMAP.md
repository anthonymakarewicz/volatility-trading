# Roadmap

This roadmap tracks committed development priorities for the backtesting library.
Exploratory ideas stay in `notes/` until promoted here.

## Now (next 1-2 milestones)

### Richer exits (requested priority)
- Extend the shipped structure-level `pnl_per_contract` stop-loss /
  take-profit exits with more advanced calibration bases such as
  `entry_risk_multiple`.
- Add half-life / signal-decay exits.

### Backtest run workflows
- Build on the current single-run workflow runner with a multi-run experiment
  layer for parameter grids, strategy comparisons, and scenario sweeps outside
  notebooks/examples.
- Persist experiment-level manifests and aggregate comparison tables alongside
  single-run report bundles.
- Expose a stable subset of preset-owned richer exit knobs in workflow YAML,
  starting with `stop_loss_pnl_per_contract` and
  `take_profit_pnl_per_contract`, while keeping advanced/custom exit-rule
  composition in Python.
- Add richer modeling support to workflow YAML only when a concrete pricing/risk
  workflow requires it; keep the current runner slice intentionally narrow until
  then.

### Diagnostics and reporting baseline
- Add core risk dashboards: margin usage timeline, call events timeline, and rolling risk-adjusted metrics.
- Add benchmark comparison tables (strategy vs volatility benchmarks) in reporting outputs.
- Add comparative performance tables for each configured benchmark, not only the strategy, with relative-performance metrics recorded alongside strategy results.
- Extend reporting to plot equity curve, drawdown, rolling Sharpe, and rolling annualized volatility for the strategy and all configured benchmarks.
- Add P&L decomposition usign transaction costs form option and hedging instrument.

## Mid-term (priority)

### Margin domain cleanup
- Unify position-level initial margin models (`options/risk/margin.py`) with account-level maintenance/liquidation logic (`backtesting/margin.py`) under clearer shared contracts.
- Separate margin model, maintenance policy, and account/liquidation policy responsibilities more explicitly before portfolio/runtime expansion.

### Position sizing v2
- Add hard Greek-aware sizing constraints (vega/gamma limits).
- Add feature-aware sizing (e.g., VIX buckets, signal z-score buckets).

### Experiment infrastructure
- Add a multi-run experiment layer for parameter grids, strategy comparisons, and execution/margin scenario sweeps.
- Persist experiment-level manifests and aggregate comparison tables alongside single-run report bundles.

### Strategy diversification
- Add a third distinct strategy family beyond `vrp_harvesting` and
  `skew_mispricing` so the current engine abstractions are pressure-tested
  outside the first two short-vol / skew-oriented paths.
- Use the next strategy path to validate that sizing, margin, and lifecycle
  contracts remain generic as strategy structure requirements diversify.

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

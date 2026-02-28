# Roadmap

This roadmap tracks committed development priorities for the backtesting library.
Exploratory ideas stay in `notes/` until promoted here.

## Now (next 1-2 milestones)

### Data adapters (next step)
- Define a canonical options-chain schema contract for backtesting inputs (source-agnostic column names and units).
- Add an adapter boundary before plan compilation (`normalize -> validate -> run`) with explicit schema errors.
- Implement built-in adapters for ORATS mapping, yfinance best-effort normalization, and generic user column maps.
- Add strict schema validation tests and adapter fixture tests for ingestion robustness.
- Document supported schemas and adapter usage in backtesting reference docs.

### Execution realism
- Add `OptionExecutionModel` for option-leg fills (slippage/fees) to mirror hedge execution modularity.
- Split P&L attribution into market P&L vs transaction costs.
- Add trade-level reason codes across entry/exit/risk transitions.

### Validation and release hygiene
- Add `CHANGELOG.md` and start tagged `0.x.y` releases.
- Expand CI scope for lint/typecheck/test consistency across all active subpackages.
- Add repeatable E2E scripts for extract/build/qc + backtest smoke runs.

### Diagnostics and reporting baseline
- Add core risk dashboards: margin usage timeline, call events timeline, and rolling risk-adjusted metrics.
- Add benchmark comparison tables (strategy vs volatility benchmarks) in reporting outputs.

## Mid-term (priority)

### Richer exits (requested priority)
- Add stop-loss / take-profit exits at structure level.
- Add optional leg-level stop/target exits.
- Add time-stop + volatility mean-reversion exits.
- Add half-life exits.
- Add event exits (earnings and macro/event calendars).

### Validation workflows
- Add walk-forward workflow with rolling recalibration/retrain windows.
- Add out-of-sample scorecards by regime and by year.
- Add parameter stability workflows (sensitivity grids/heatmaps).
- Add reality-check workflows with permutation/bootstrap style tests on trade sequencing.
- Add reality-check workflows with spot/IV noise perturbation and recomputed equity-curve stress tests.

### Position sizing v2
- Add hard Greek-aware sizing constraints (vega/gamma limits).
- Add feature-aware sizing (e.g., VIX buckets, signal z-score buckets).

## Later (long-term)

### Portfolio and risk controls
- Move from single-position loop to position-book runtime (multiple concurrent positions).
- Add strategy-level exposure caps first; portfolio-level caps after multi-position core is stable.
- Add drawdown kill-switch and cool-down controls.

### Architecture evolution
- Introduce event-driven runtime compatible with both backtest and paper-trading simulation.
- Keep shared domain contracts stable while evolving execution engines.

### UX and publishing
- Promote stable plotting/report helpers to documented public API.
- Improve GitHub Pages presentation and publication pipeline.

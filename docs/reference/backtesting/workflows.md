# Backtest Runner Workflows

This page is the source of truth for the YAML workflow schema consumed by
`backtest-run`. It also points to the two shipped configs that should be
treated as the canonical starting templates.

## Command

Run one workflow with:

```bash
backtest-run --config config/backtesting/vrp_harvesting.yml
```

Validate config parsing and local data assembly without executing the backtest:

```bash
backtest-run --config config/backtesting/vrp_harvesting.yml --dry-run
```

Common top-level CLI overrides:

- `--ticker`
- `--start`
- `--end`
- `--output-root`
- `--run-id`

## Canonical Templates

Use these files as the starting point for new runner configs:

- [`config/backtesting/vrp_harvesting.yml`](../../../config/backtesting/vrp_harvesting.yml)
  - generic baseline VRP harvesting workflow
  - explicit strategy, execution, and margin-model configuration
  - minimal single-run template for `backtest-run`
- [`config/backtesting/skew_mispricing.yml`](../../../config/backtesting/skew_mispricing.yml)
  - richer skew workflow
  - ORATS daily features
  - yfinance hedge and benchmark series
  - FRED risk-free-rate series
  - explicit `broker.margin.policy` example using rate-sourced financing

Repo-style relative paths inside workflow configs, such as `data/...` and
`reports/...`, are resolved from the current working directory first and then
fall back to the repository root. That allows the shipped configs to work from
notebook subdirectories as well as from the repo root.

## Top-Level YAML Shape

The current workflow config supports these top-level sections:

- `logging`
- `data`
- `strategy`
- `account`
- `execution`
- `broker`
- `modeling`
- `run`
- `reporting`

Notes:

- `logging` is consumed by the CLI app layer, not by the internal runner parser.
- `dry_run` is a CLI/runtime flag and is normally set through `--dry-run`, not
  in YAML.
- `broker` currently supports only margin configuration.
- `modeling` remains intentionally narrow in the current runner slice and
  currently supports scenario-generator configuration only.

## Section Reference

### `logging`

CLI logging configuration.

Supported keys:

- `level`
- `format`
- `file`
- `color`

### `data`

Input market-data sources for one run.

Supported subsections:

- `options` (required)
- `features` (optional)
- `hedge` (optional)
- `benchmark` (optional)
- `rates` (optional)

#### `data.options`

Required keys:

- `ticker`

Supported keys:

- `ticker`
- `provider`
- `proc_root`
- `symbol`
- `default_contract_multiplier`
- `dte_min`
- `dte_max`

Current support:

- `provider: orats`

#### `data.features`

Required keys:

- `ticker`

Supported keys:

- `ticker`
- `provider`
- `proc_root`

Current support:

- `provider: orats`

#### `data.hedge` / `data.benchmark`

Required keys:

- `ticker`

Supported keys:

- `ticker`
- `provider`
- `proc_root`
- `price_column`
- `symbol`
- `contract_multiplier`

Current support:

- `provider: yfinance`

#### `data.rates`

Supported keys:

- `provider`
- `constant_rate`
- `series_id`
- `column`
- `proc_root`

Current support:

- `provider: constant`
- `provider: fred`

Required when `data.rates` is present:

- `provider`
- `constant_rate` for `provider: constant`
- `series_id` for `provider: fred`

`proc_root` note for `provider: fred`:

- the runner accepts either the FRED source root (`data/processed/fred`) or the
  rates-domain root (`data/processed/fred/rates`)

### `strategy`

Named strategy preset plus preset parameters.

Supported keys:

- `name` (required)
- `params`
- `signal`

Notes:

- `strategy.name` must be provided explicitly in YAML or via a merged config
  layer. `backtest-run` does not assume a default strategy preset.

Current preset names:

- `vrp_harvesting`
- `skew_mispricing`

Signal note:

- `vrp_harvesting` defaults to `short_only` when the YAML omits `signal`.
- `skew_mispricing` can omit `signal` and use its preset default.

Current signal names:

- `short_only`
- `long_only`
- `zscore`

#### `strategy.params`

`strategy.params` is preset-specific. Unknown keys are rejected by the runner.

The current stable runner YAML surface covers the scalar / date-window /
boolean-style preset knobs below.

`vrp_harvesting`:

- `rebalance_period`
- `max_holding_period`
- `allow_same_day_reentry_on_rebalance`
- `allow_same_day_reentry_on_max_holding`
- `allow_same_day_reentry_on_stop_loss`
- `allow_same_day_reentry_on_take_profit`
- `target_dte`
- `max_dte_diff`
- `risk_budget_pct`
- `margin_budget_pct`
- `stop_loss_pnl_per_contract`
- `take_profit_pnl_per_contract`
- `min_contracts`
- `max_contracts`
- `entry_risk_basis`
- `delta_hedge`

`skew_mispricing`:

- `rebalance_period`
- `max_holding_period`
- `allow_same_day_reentry_on_rebalance`
- `allow_same_day_reentry_on_max_holding`
- `allow_same_day_reentry_on_stop_loss`
- `allow_same_day_reentry_on_take_profit`
- `target_dte`
- `max_dte_diff`
- `delta_target_abs`
- `delta_tolerance`
- `risk_budget_pct`
- `margin_budget_pct`
- `stop_loss_pnl_per_contract`
- `take_profit_pnl_per_contract`
- `min_contracts`
- `max_contracts`
- `entry_risk_basis`
- `delta_hedge`

Current YAML boundary notes:

- `stop_loss_pnl_per_contract` and `take_profit_pnl_per_contract` are the
  current runner-owned richer-exit knobs for built-in presets.
- `delta_hedge` is supported as a nested preset param for built-in strategies.
  Use the same shape as the Python `DeltaHedgePolicy` contract:
  `enabled`, `target_net_delta`, `trigger`, `rebalance_to`,
  `allow_missing_hedge_price`, `min_rebalance_qty`, and `max_rebalance_qty`.
- `delta_hedge.trigger.band_model` accepts:
  - `model: fixed` with `half_width_abs`
  - `model: ww` with `WWDeltaBandModel` params
- More advanced preset fields such as `filters`, `exit_rule_set`, and
  `reentry_policy` remain part of the Python library path, not the documented
  runner YAML surface.
- When a param is omitted, the preset's Python default still applies.

### `account`

Run-level capital configuration.

Supported keys:

- `initial_capital`

### `execution`

Run-level execution models.

Supported subsections:

- `option`
- `hedge`

#### `execution.option`

Supported keys:

- `model`
- `params`

Current model names:

- `bid_ask_fee`
- `mid_no_cost`

#### `execution.hedge`

Supported keys:

- `model`
- `params`

Current model names:

- `fixed_bps`
- `mid_no_cost`

### `broker`

Broker and margin configuration.

Current support:

- `margin`

#### `broker.margin`

Supported keys:

- `model`
- `policy`

#### `broker.margin.model`

Supported keys:

- `name` (required)
- `params`

Current model names:

- `regt`
- `portfolio_margin_proxy`

#### `broker.margin.policy`

Supported keys:

- `maintenance_margin_ratio`
- `margin_call_grace_days`
- `liquidation_mode`
- `liquidation_buffer_ratio`
- `apply_financing`
- `cash_rate_annual`
- `cash_rate_source`
- `borrow_rate_annual`
- `borrow_rate_spread`
- `trading_days_per_year`

Template note:

- `config/backtesting/skew_mispricing.yml` is the canonical example showing an
  explicit `broker.margin.policy` block with `cash_rate_source: data_rates`
  and `borrow_rate_spread`.

Financing note:

- scalar financing still works through `cash_rate_annual` and
  `borrow_rate_annual`
- rate-sourced financing uses:
  - `cash_rate_source: data_rates`
  - optional `borrow_rate_spread`
- `cash_rate_source: data_rates` requires `data.rates`

### `modeling`

Runtime pricing/risk hooks used by sizing and PM-style margin.

Current support:

- `scenario_generator`

#### `modeling.scenario_generator`

Supported keys:

- `name` (required)
- `params`

Current supported names:

- `fixed_grid`

`fixed_grid` params currently support:

- `spot_shocks_pct`
- `vol_shocks`
- `risk_reversal_shocks`
- `rate_shocks`
- `time_shocks_years`
- `deduplicate`

Example:

```yaml
modeling:
  scenario_generator:
    name: fixed_grid
    params:
      risk_reversal_shocks: [-0.05, 0.0, 0.05]
```

Notes:

- `risk_reversal_shocks` is optional and defaults to `[0.0]`, preserving the
  previous spot + parallel-vol stress behavior.
- positive RR shock uses call-minus-put convention:
  - call wing up
  - put wing down
- RR shocks affect stress-based sizing and PM-style margin through the shared
  scenario-generator / risk-estimator stack.

### `run`

Backtest date window.

Supported keys:

- `start_date`
- `end_date`

### `reporting`

Report bundle output and labeling.

Supported keys:

- `output_root`
- `run_id`
- `benchmark_name`
- `include_dashboard_plot`
- `include_component_plots`
- `save_report_bundle`

`run_id` guidance:

- `run_id` should identify the specific run, not repeat the strategy name.
- Report bundles already save under `reports/backtests/<strategy>/<run_id>`.
- Prefer values such as `spy_2018_2020` over `vrp_harvesting_spy`.

Example:

```yaml
reporting:
  include_dashboard_plot: false
  include_component_plots: true
```

This keeps the main dashboard off while persisting the component plots under
`plots/`.

## Output Layout

When `reporting.save_report_bundle: true`, the runner writes one report bundle
under:

```text
<output_root>/<strategy_name>/<run_id>/
```

With the default output root, this is typically:

```text
reports/backtests/<strategy_name>/<run_id>/
```

Stable artifacts written by the current reporting writer:

- `manifest.json`
  - artifact index for the run
  - includes report metadata and the relative paths of persisted outputs
- `run_config.json`
  - resolved workflow/report payload used for the run
  - includes both workflow config and lightweight resolved metadata
- `summary_metrics.json`
  - headline performance metrics for the completed run
- `equity_and_drawdown.csv`
  - daily strategy equity/drawdown table
  - includes benchmark equity/drawdown columns when a benchmark is configured
- `trades.csv`
  - realized trade-level output for the run
  - includes sizing diagnostics such as raw risk-budget and margin-budget
    contract counts plus the binding sizing constraint for each realized trade
- `exposures_daily.csv`
  - daily exposure and risk snapshot table
  - may include explicit factor-model exposure columns such as
    `factor_exposure_<name>` when a strategy attaches a factor model
- `margin_diagnostics_daily.csv`
  - daily margin-account state and financing/hedging diagnostics
  - includes margin requirements, utilization, margin-call state, and
    forced-liquidation event columns
- `rolling_metrics.csv`
  - rolling 63-day strategy diagnostics
  - includes rolling return, annualized volatility, Sharpe, and drawdown
  - includes benchmark rolling columns and relative equity spread when a
    benchmark is configured
- `pnl_attribution_daily.csv`
  - daily Greek-attribution P&L table
  - includes `delta_pnl`, `Delta_PnL`, `Unhedged_Delta_PnL`, `Gamma_PnL`,
    `Vega_PnL`, `Theta_PnL`, and `Other_PnL`
  - may include additional factor-model attribution columns such as
    `IV_Level_PnL` and `RR_Skew_PnL` when the strategy provides an explicit
    factor decomposition model
- `benchmark_comparison.json`
  - written only when a benchmark is configured
  - records nested strategy, benchmark, and relative summary metrics

Plot outputs are written only when figures are enabled:

- `plots/performance_dashboard.png`
  - written when `include_dashboard_plot: true`
- `plots/equity_vs_benchmark.png`
- `plots/drawdown.png`
- `plots/greeks_exposure.png`
- `plots/margin_account.png`
- `plots/rolling_metrics.png`
- `plots/pnl_attribution.png`
  - written when `include_component_plots: true`

Notes:

- `manifest.json` is the best entrypoint for downstream automation because it
  records the artifact set actually written for that run.
- If `reporting.save_report_bundle: false`, the backtest still runs, but these
  files are not persisted to disk.

## Recommended Starting Point

Use this sequence for a first run:

1. Copy `config/backtesting/vrp_harvesting.yml`.
2. Adjust `data.options.proc_root` to your processed ORATS options-chain root.
3. Run `backtest-run --config ... --dry-run`.
4. Fix any missing local datasets.
5. Remove `--dry-run` and execute the workflow.

If you need daily features, benchmark comparison, or FRED financing input,
start from `config/backtesting/skew_mispricing.yml` instead.

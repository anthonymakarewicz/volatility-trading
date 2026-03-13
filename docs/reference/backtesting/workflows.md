# Backtest Runner Workflows

This page documents the YAML workflow schema consumed by `backtest-run` and
points to the two shipped configs that should be treated as the canonical
starting templates.

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
  - minimal VRP harvesting workflow
  - constant risk-free rate
  - no extra features, hedge market, or benchmark inputs
- [`config/backtesting/skew_mispricing.yml`](../../../config/backtesting/skew_mispricing.yml)
  - richer skew workflow
  - ORATS daily features
  - yfinance hedge and benchmark series
  - FRED risk-free-rate series

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
  currently supports only an omitted or empty mapping.

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
- `adapter_name`
- `symbol`
- `default_contract_multiplier`

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

`proc_root` note for `provider: fred`:

- the runner accepts either the FRED source root (`data/processed/fred`) or the
  rates-domain root (`data/processed/fred/rates`)

### `strategy`

Named strategy preset plus preset parameters.

Supported keys:

- `name` (required)
- `params`
- `signal`

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
- `borrow_rate_annual`
- `trading_days_per_year`

### `modeling`

Reserved for future runner expansion.

Current support:

- empty mapping only

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

## Recommended Starting Point

Use this sequence for a first run:

1. Copy `config/backtesting/vrp_harvesting.yml`.
2. Adjust `data.options.proc_root` to your processed ORATS options-chain root.
3. Run `backtest-run --config ... --dry-run`.
4. Fix any missing local datasets.
5. Remove `--dry-run` and execute the workflow.

If you need daily features, benchmark comparison, or FRED financing input,
start from `config/backtesting/skew_mispricing.yml` instead.

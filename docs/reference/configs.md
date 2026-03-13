# Configs

This page indexes the YAML config families used across ETL and backtesting
workflows.

Config precedence is:

```text
CLI overrides > YAML > app defaults
```

Use:

- `--print-config` to inspect the merged config payload
- `--dry-run` when the command supports validation without full execution

Detailed command usage lives in
[entrypoints.md](entrypoints.md).

## Config Families

Current config directories under `config/`:

- `config/orats/`
- `config/fred/`
- `config/yfinance/`
- `config/optionsdx/`
- `config/backtesting/`

## ETL Configs

ETL configs are provider- or pipeline-oriented and are consumed by data-ingest,
extract, build, QC, or sync commands.

### ORATS

Files:

- `config/orats/api_download.yml`
- `config/orats/api_extract.yml`
- `config/orats/ftp_download.yml`
- `config/orats/ftp_extract.yml`
- `config/orats/options_chain_build.yml`
- `config/orats/daily_features_build.yml`
- `config/orats/qc_options_chain.yml`
- `config/orats/qc_daily_features.yml`

Typical schema:

- `paths.raw_root`
- `paths.inter_root`
- `paths.proc_root`
- `paths.monies_implied_root` for options-chain build

Use these for:

- raw ORATS download/extract
- processed options-chain and daily-features build
- processed dataset QC

### FRED

Files:

- `config/fred/sync.yml`

Typical schema:

- `paths.raw_root`
- `paths.proc_root`
- `fred.*` credentials or token env
- domain/series selection

Use this for:

- rates/time-series sync from FRED into processed datasets

### yfinance

Files:

- `config/yfinance/time_series_sync.yml`

Typical schema:

- `paths.raw_root`
- `paths.proc_root`
- ticker list
- date range and interval controls

Use this for:

- benchmark, hedge, or auxiliary market-series sync

### OptionsDX

Files:

- `config/optionsdx/prepare_panel.yml`

Typical schema:

- `paths.raw_root`
- `paths.proc_root`
- ticker/year selection and panel-build controls

Use this for:

- preparing processed OptionsDX panels from raw archives

## Backtesting Workflow Configs

Backtesting configs are strategy/workflow-oriented rather than provider-oriented.

Files:

- `config/backtesting/vrp_harvesting.yml`
- `config/backtesting/skew_mispricing.yml`

These are consumed by:

- `backtest-run`

Use these for:

- one complete config-driven backtest run
- strategy preset selection
- dataset source composition
- execution, margin, and report settings

Detailed schema, canonical templates, and output-layout docs live in:

- [Backtest Runner Workflows](backtesting/workflows.md)

## Choosing the Right Config

Use ETL configs when the goal is to build or refresh datasets.

Use backtesting configs when the goal is to run a strategy against already
prepared datasets and save a report bundle.

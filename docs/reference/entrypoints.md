# Entrypoints

This project exposes console scripts after installation. Run:
```bash
uv pip install -e .
```

`pip install -e .` remains a supported fallback.

If you add or change `project.scripts` in `pyproject.toml`, reinstall to
regenerate console scripts:

```bash
uv pip install -e .
rehash  # zsh: refresh command lookup
```

## ORATS

Commands:
- `orats-api-download` (config: `config/orats/api_download.yml`)
- `orats-api-extract` (config: `config/orats/api_extract.yml`)
- `orats-ftp-download` (config: `config/orats/ftp_download.yml`)
- `orats-ftp-extract` (config: `config/orats/ftp_extract.yml`)
- `orats-build-options-chain` (config: `config/orats/options_chain_build.yml`)
- `orats-build-daily-features` (config: `config/orats/daily_features_build.yml`)
- `orats-qc-daily-features` (config: `config/orats/qc_daily_features.yml`)
- `orats-qc-options-chain` (config: `config/orats/qc_options_chain.yml`)

Note:
- The current full options ETL path is provided through these ORATS commands.

Example:
```bash
orats-api-download --config config/orats/api_download.yml
```

## Backtesting

Commands:
- `backtest-run` (configs: `config/backtesting/vrp_harvesting.yml`,
  `config/backtesting/skew_mispricing.yml`)

Example:
```bash
backtest-run --config config/backtesting/vrp_harvesting.yml --dry-run
backtest-run --config config/backtesting/skew_mispricing.yml
```

Notes:
- Runs one config-driven backtest workflow per invocation.
- Strategy selection is explicit; workflow YAML must provide `strategy.name`.
- Supports top-level CLI overrides for:
  - `--ticker`
  - `--start`
  - `--end`
  - `--output-root`
  - `--run-id`
- `--dry-run` validates config parsing and local data assembly without
  executing the backtest.
- The provided workflow configs show:
  - a minimal VRP harvesting run
  - a richer skew-mispricing run with daily features, benchmark, and FRED rates
- Full YAML schema and template guidance:
  [`docs/reference/backtesting/workflows.md`](backtesting/workflows.md)

## OptionsDX

Commands:
- `optionsdx-prepare-panel` (config: `config/optionsdx/prepare_panel.yml`)

Example:
```bash
optionsdx-prepare-panel --config config/optionsdx/prepare_panel.yml
```

Notes:
- Expects raw archives under `data/raw/optionsdx/<TICKER>/<YEAR>/*.7z`.
- Writes one processed panel per ticker under
  `data/processed/optionsdx/<TICKER>/`.

## Market Feeds

Commands:
- `fred-sync` (config: `config/fred/sync.yml`)
- `yfinance-sync` (config: `config/yfinance/time_series_sync.yml`)

Examples:
```bash
fred-sync --config config/fred/sync.yml
yfinance-sync --config config/yfinance/time_series_sync.yml
```

## Common Flags

- `--config <path>`: load a YAML config file.
- `--print-config`: print the merged config (JSON) and exit.
- `--dry-run`: validate config/paths/creds and log the execution plan **without**
  making network calls or writing files.

## Config Schema (Paths)

All ORATS apps now use a **consistent** paths schema:

- `paths.raw_root`
- `paths.inter_root` (intermediate data)
- `paths.proc_root` (processed data)
- `paths.monies_implied_root` (options-chain build only)

OptionsDX panel build app uses:

- `paths.raw_root`
- `paths.proc_root`

Backtest workflow configs use top-level sections instead:

- `data`
- `strategy`
- `account`
- `execution`
- `broker`
- `modeling`
- `run`
- `reporting`

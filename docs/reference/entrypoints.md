# Entrypoints

This project exposes console scripts after installation. Run:
```bash
pip install -e .
```

If you add or change `options.entry_points` in `setup.cfg`, reinstall to
regenerate console scripts:

```bash
python -m pip install -e . --no-build-isolation --no-deps
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

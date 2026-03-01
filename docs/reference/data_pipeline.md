# Data Pipeline

This project uses two pipeline patterns:

1. **ORATS staged pipeline** (raw -> intermediate -> processed -> QC)
2. **External market feed lightweight pipeline** (raw -> processed)

Current scope:

- Full options ETL is currently ORATS-first.
- OptionsDX is supported as a local raw-archive to processed-panel pipeline.
- External feed pipeline (FRED/yfinance) currently supports market/rates time series.
- Backtesting can consume non-ORATS options data via the options-engine adapter
  boundary (`normalize -> validate -> run`).

Runtime YAML configs are grouped by source under `config/`:
- `config/orats/`
- `config/optionsdx/`
- `config/fred/`
- `config/yfinance/`

## ORATS Staged Pipeline

The ORATS pipeline is staged as:

1. **Raw** downloads (API + FTP ZIPs)
2. **Intermediate** extracts (parquet partitions)
3. **Processed** datasets (options chain + daily features)
4. **QC** checks for data integrity

For pipeline-specific failure modes and fixes, see
[ORATS Troubleshooting](orats_troubleshooting.md).
For general tooling/dev issues, see [Troubleshooting](../troubleshooting.md).
For adding new daily-features columns or endpoints, see
[Daily Features Onboarding](../contributing/daily_features_onboarding.md).

### Full Runbook (All Steps)

Run in this order:

### 1) API download (raw JSON)

```bash
orats-api-download --config config/orats/api_download.yml --dry-run
orats-api-download --config config/orats/api_download.yml
```

### 2) API extract (intermediate parquet)

```bash
orats-api-extract --config config/orats/api_extract.yml --dry-run
orats-api-extract --config config/orats/api_extract.yml
```

### 3) FTP download (raw ZIP)

```bash
orats-ftp-download --config config/orats/ftp_download.yml --dry-run
orats-ftp-download --config config/orats/ftp_download.yml
```

### 4) FTP extract (intermediate parquet)

```bash
orats-ftp-extract --config config/orats/ftp_extract.yml --dry-run
orats-ftp-extract --config config/orats/ftp_extract.yml
```

### 5) Build processed options chain

```bash
orats-build-options-chain --config config/orats/options_chain_build.yml --dry-run
orats-build-options-chain --config config/orats/options_chain_build.yml
```

### 6) Build processed daily features

```bash
orats-build-daily-features --config config/orats/daily_features_build.yml --dry-run
orats-build-daily-features --config config/orats/daily_features_build.yml
```

### 7) QC options chain

```bash
orats-qc-options-chain --config config/orats/qc_options_chain.yml --dry-run
orats-qc-options-chain --config config/orats/qc_options_chain.yml
```

### 8) QC daily features

```bash
orats-qc-daily-features --config config/orats/qc_daily_features.yml --dry-run
orats-qc-daily-features --config config/orats/qc_daily_features.yml
```

### High-Level Flow

```text
ORATS API (raw JSON)
    └─ orats-api-download
    └─ orats-api-extract

ORATS FTP (raw ZIP)
    └─ orats-ftp-download
    └─ orats-ftp-extract

Processed datasets
    └─ orats-build-options-chain
    └─ orats-build-daily-features

Quality Control
    └─ orats-qc-options-chain
    └─ orats-qc-daily-features
```

### Directory Layout (by stage)

```
data/
  raw/
    orats/
      api/
      ftp/
  intermediate/
    orats/
      api/
      ftp/strikes/
  processed/
    orats/
      options_chain/
      daily_features/
```

## External Market Feed Lightweight Pipeline

For stable API sources like FRED and yfinance, keep a lightweight pattern:

1. **Raw** source snapshots by domain
2. **Processed** analysis-ready parquet tables

For yfinance index-like symbols, use clean ticker names in config
(`SP500TR`, `VIX`). The sync step maps to caret-prefixed Yahoo symbols
internally when required and stores caret-free tickers in outputs.

Current CLI entrypoints:

```bash
fred-sync --config config/fred/sync.yml --dry-run
fred-sync --config config/fred/sync.yml

yfinance-sync --config config/yfinance/time_series_sync.yml --dry-run
yfinance-sync --config config/yfinance/time_series_sync.yml
```

Directory layout:

```text
data/
  raw/
    fred/
      rates/
      market/
    yfinance/
      time_series/
  processed/
    fred/
      rates/
      market/
    yfinance/
      time_series/
```

Design notes:
- Read from `processed/` by default in backtests/research.
- Use `raw/` for lineage/debug/rebuild workflows.
- Keep external feeds in `raw -> processed` unless complexity justifies an
  explicit intermediate stage.

## OptionsDX Lightweight Pipeline

OptionsDX currently uses a lightweight local-ingestion pattern:

1. Download yearly `.7z` archives manually from OptionsDX.
2. Store raw files by ticker and year.
3. Build one processed panel per ticker.

CLI entrypoint:

```bash
optionsdx-prepare-panel --config config/optionsdx/prepare_panel.yml --dry-run
optionsdx-prepare-panel --config config/optionsdx/prepare_panel.yml
```

Directory layout:

```text
data/
  raw/
    optionsdx/
      SPY/
        2010/
          *.7z
        2011/
          *.7z
  processed/
    optionsdx/
      SPY/
        optionsdx_panel_2010_2023.parquet
```

For provider download steps, see [OptionsDX Setup](optionsdx_setup.md).
For canonical options schema and adapter usage, see
[Options Data Adapters](options_data_adapters.md).

## Tips

- Use `--dry-run` to validate config/paths/creds before executing.
- Use `--print-config` to see the final merged config.
- QC exits non-zero if a dataset fails checks (useful for cron/CI).
- Install console scripts first if needed: `pip install -e .`

# Entrypoints

This project exposes console scripts after installation. Run:
```bash
pip install -e .
```

## ORATS

Commands:
- `orats-api-download` (config: `config/orats_api_download.yml`)
- `orats-api-extract` (config: `config/orats_api_extract.yml`)
- `orats-ftp-download` (config: `config/orats_ftp_download.yml`)
- `orats-ftp-extract` (config: `config/orats_ftp_extract.yml`)
- `orats-build-options-chain` (config: `config/orats_options_chain_build.yml`)
- `orats-build-daily-features` (config: `config/orats_daily_features_build.yml`)
- `orats-qc-daily-features` (config: `config/orats_qc_daily_features.yml`)
- `orats-qc-options-chain` (config: `config/orats_qc_options_chain.yml`)

Example:
```bash
orats-api-download --config config/orats_api_download.yml
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

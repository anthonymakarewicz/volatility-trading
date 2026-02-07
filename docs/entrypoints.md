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

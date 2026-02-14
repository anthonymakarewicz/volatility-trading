# Configs

All ORATS apps accept a YAML config via `--config`. The merged config is:

```
CLI overrides > YAML > defaults
```

Use `--print-config` to see the final JSON config and `--dry-run` to validate
inputs without executing.

## Common Path Keys

All configs use a consistent paths schema:

- `paths.raw_root`
- `paths.inter_root`
- `paths.proc_root`
- `paths.monies_implied_root` (options-chain build only)

## ORATS API Download

File: `config/orats_api_download.yml`

Key fields:

- `endpoint` (string)
- `tickers` (list)
- `year_whitelist` (list or null)
- `fields` (list or null)
- `compression` ("gz" or "none")
- `overwrite` (bool)
- `sleep_s` (float)
- `fail_on_failed` (bool)

## ORATS API Extract

File: `config/orats_api_extract.yml`

Key fields:

- `paths.raw_root`
- `paths.inter_root`
- `endpoint`
- `tickers`
- `year_whitelist`
- `raw_compression`
- `overwrite`
- `parquet_compression`
- `fail_on_failed`

## ORATS FTP Download

File: `config/orats_ftp_download.yml`

Key fields:

- `paths.raw_root`
- `year_whitelist` (list or null)
- `validate_zip` (bool)
- `max_workers` (int)
- `fail_on_failed` (bool)
- `ftp.user_env` / `ftp.pass_env`
- `ftp.user` / `ftp.password`
- `ftp.host` / `ftp.remote_base_dirs` (optional)

## ORATS FTP Extract

File: `config/orats_ftp_extract.yml`

Key fields:

- `paths.raw_root`
- `paths.inter_root`
- `tickers`
- `year_whitelist`
- `strict`

## Options Chain Build

File: `config/orats_options_chain_build.yml`

Key fields:

- `paths.inter_root` (FTP strikes)
- `paths.monies_implied_root` (API)
- `paths.proc_root`
- `tickers`
- `years`
- `dte_min`, `dte_max`
- `moneyness_min`, `moneyness_max`
- `collect_stats`, `derive_put_greeks`, `merge_dividend_yield`
- `columns` (optional override)

## Daily Features Build

File: `config/orats_daily_features_build.yml`

Key fields:

- `paths.inter_root`
- `paths.proc_root`
- `tickers`
- `endpoints`
- `prefix_endpoint_cols`
- `priority_endpoints`
- `collect_stats`
- `columns` (optional override)

## QC (Options Chain / Daily Features)

Files:

- `config/orats_qc_options_chain.yml`
- `config/orats_qc_daily_features.yml`

Key fields:

- `paths.proc_root`
- `tickers`
- `write_json`
- `out_json`

Options-chain QC also includes:

- `dte_bins`, `delta_bins`
- `roi_dte_min`, `roi_dte_max`
- `roi_delta_min`, `roi_delta_max`
- `top_k_buckets`

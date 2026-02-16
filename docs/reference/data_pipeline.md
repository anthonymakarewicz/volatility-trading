# Data Pipeline

This project uses a staged ORATS pipeline:

1. **Raw** downloads (API + FTP ZIPs)
2. **Intermediate** extracts (parquet partitions)
3. **Processed** datasets (options chain + daily features)
4. **QC** checks for data integrity

For pipeline-specific failure modes and fixes, see
[ORATS Troubleshooting](orats_troubleshooting.md).
For general tooling/dev issues, see [Troubleshooting](../troubleshooting.md).
For adding new daily-features columns or endpoints, see
[Daily Features Onboarding](../contributing/daily_features_onboarding.md).

## Full Runbook (All Steps)

Run in this order:

### 1) API download (raw JSON)

```bash
orats-api-download --config config/orats_api_download.yml --dry-run
orats-api-download --config config/orats_api_download.yml
```

### 2) API extract (intermediate parquet)

```bash
orats-api-extract --config config/orats_api_extract.yml --dry-run
orats-api-extract --config config/orats_api_extract.yml
```

### 3) FTP download (raw ZIP)

```bash
orats-ftp-download --config config/orats_ftp_download.yml --dry-run
orats-ftp-download --config config/orats_ftp_download.yml
```

### 4) FTP extract (intermediate parquet)

```bash
orats-ftp-extract --config config/orats_ftp_extract.yml --dry-run
orats-ftp-extract --config config/orats_ftp_extract.yml
```

### 5) Build processed options chain

```bash
orats-build-options-chain --config config/orats_options_chain_build.yml --dry-run
orats-build-options-chain --config config/orats_options_chain_build.yml
```

### 6) Build processed daily features

```bash
orats-build-daily-features --config config/orats_daily_features_build.yml --dry-run
orats-build-daily-features --config config/orats_daily_features_build.yml
```

### 7) QC options chain

```bash
orats-qc-options-chain --config config/orats_qc_options_chain.yml --dry-run
orats-qc-options-chain --config config/orats_qc_options_chain.yml
```

### 8) QC daily features

```bash
orats-qc-daily-features --config config/orats_qc_daily_features.yml --dry-run
orats-qc-daily-features --config config/orats_qc_daily_features.yml
```

## High-Level Flow

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

## Directory Layout (by stage)

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

## Tips

- Use `--dry-run` to validate config/paths/creds before executing.
- Use `--print-config` to see the final merged config.
- QC exits non-zero if a dataset fails checks (useful for cron/CI).
- Install console scripts first if needed: `pip install -e .`

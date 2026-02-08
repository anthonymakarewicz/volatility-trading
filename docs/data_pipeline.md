# Data Pipeline

This project uses a staged ORATS pipeline:

1. **Raw** downloads (API + FTP ZIPs)
2. **Intermediate** extracts (parquet partitions)
3. **Processed** datasets (options chain + daily features)
4. **QC** checks for data integrity

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


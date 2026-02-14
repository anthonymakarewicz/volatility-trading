# ORATS Troubleshooting

Use this page for ORATS ETL/QC pipeline-specific failures.
For repo-wide tooling/dev issues, see
[Troubleshooting](../troubleshooting.md).

## Missing ORATS Credentials

- Symptom: API/FTP apps fail early with missing token or credentials.
- Likely cause: env vars are not set in active shell.
- Fix:
  - set `ORATS_API_KEY` for API apps
  - set `ORATS_FTP_USER` and `ORATS_FTP_PASS` for FTP apps
  - verify `.env` is loaded in your execution context

## Config Key Errors

- Symptom: config validation fails due to missing path keys.
- Likely cause: YAML still uses an old schema.
- Fix: ensure config includes the expected keys:
  - `paths.raw_root`
  - `paths.inter_root`
  - `paths.proc_root`
  - `paths.monies_implied_root` (options chain only)
- Related docs:
  - [Configs](configs.md)
  - [Entrypoints](entrypoints.md)

## Empty Output or No Jobs

- Symptom: run completes but no files or very few files are produced.
- Likely cause:
  - no matching endpoint/ticker/date jobs
  - FTP year filter too restrictive
  - missing raw snapshots for extract stages
- Fix:
  - run with `--dry-run` to inspect planned work
  - validate date range and endpoint/ticker filters
  - for FTP, check `year_whitelist`
  - for API extract, confirm raw inputs exist for requested range
- Related docs:
  - [Data Pipeline](data_pipeline.md)

## QC Returns Non-zero Exit Code

- Symptom: QC command exits non-zero even when process appears to finish.
- Likely cause: one or more tickers failed configured QC checks.
- Fix:
  - inspect QC artifacts when `write_json=true`:
    - `qc_summary.json`
    - `qc_config.json`
  - review failed check names and thresholds
  - rerun on a narrower ticker/date slice for debugging
- Related docs:
  - [Data Pipeline](data_pipeline.md)

## Path Layout Mismatch

- Symptom: downstream stage cannot find upstream outputs.
- Likely cause: inconsistent `paths.*` roots across stage configs.
- Fix:
  - align `raw_root`, `inter_root`, and `proc_root` across all configs
  - print effective config with `--print-config` before writes
- Related docs:
  - [Data Pipeline](data_pipeline.md)
  - [Configs](configs.md)

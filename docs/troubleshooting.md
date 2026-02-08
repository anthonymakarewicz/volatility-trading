# Troubleshooting

## Credentials Missing

**API download/extract**

- Error: `Missing ORATS API token`
- Fix: set `ORATS_API_KEY` in your environment or `.env`

**FTP download**

- Error: `Missing ORATS FTP credentials`
- Fix: set `ORATS_FTP_USER` and `ORATS_FTP_PASS`

## Config Key Errors

- If a path is missing, make sure your YAML uses the new schema:
  - `paths.raw_root`
  - `paths.inter_root`
  - `paths.proc_root`
  - `paths.monies_implied_root` (options-chain only)

## QC Exits Non-Zero

- QC apps return a non-zero exit code if **any ticker fails**.
- Inspect JSON output if `write_json=true`:
  - `qc_summary.json`
  - `qc_config.json`

## Empty Output / No Jobs

- FTP download: check `year_whitelist` or confirm remote directories.
- API extract: ensure raw snapshots exist for the endpoint/year range.

## Pytest Import Errors

- Make sure `pyproject.toml` includes:
  - `testpaths = ["tests"]`
  - `pythonpath = ["src"]`
- Avoid duplicate test filenames (`test_download.py`) across directories.


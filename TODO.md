## ORATS API Download & Extract
- Refactor download/extract into sub modules like this:

```plaintext
etl/orats/api/
  __init__.py
  api.py              # stable public entrypoints
  client.py           # OratsClient / http details
  endpoints.py        # endpoint specs / mapping
  download.py         # orchestration for API downloads
  extract.py          # orchestration for API extracts (if needed)
  io.py               # disk layout helpers (paths, read/write)
  types.py            # small shared dataclasses/enums (optional)
  helpers.py          # tiny shared helpers (optional, only if truly shared)
```

- Move constants like MAX_PER_CALL into config or constants at root

## ORATS Processed
- Refactor processed further into modules (steps)
- Remove _ for imported functions

## Daily Features Processed:
- Implement processed/daily_features

- Try makign heleprs in processed/otpiosn_chain used in daily_fetaures into a 
processed/common.py

## Quality Checks:
- Make the top n violaitons as part of Spec for Soft
- Split package strcuture by QC name (options_chain vs daily_features) like this:

```plaintext
etl/orats/qc/
  __init__.py
  api.py                  # tiny stable public entrypoints
  types.py
  runners.py
  report.py
  serialization.py
  summarizers.py          # truly shared summarizers only (rare)
  common/                 # optional; only if this grows
    __init__.py
    manifest.py           # read_exercise_style etc if reused
    roi.py                # apply_roi_filter if reused
  suites/
    __init__.py
    options_chain/
      __init__.py
      run.py              # run_qc for options chain (or runner.py)
      _run_helpers.py     # helpers for run_qc
      hard/
      soft/
      info/
    daily_features/
      __init__.py
      run.py
      hard/
      soft/
      info/
```

- Implement qc/suites/daily_features

## Unit tests
- Add unit tests

## Scripts
- Create e2e scripts that download, extract and processed, or maybe pass the steps
to be performed as arg
- Maybe add a CLI script too

## Documentation
- Add module docstring for eahc file
- Add full docstrings for public API and shrot ones for internal helpers
- Add a doc for the ETL ORATS module like a graph with each step
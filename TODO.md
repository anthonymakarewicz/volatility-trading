## Scripts
- Add --dry-run everywhere:
    * Validate config, paths, creds, and log “what would run” without writing anything.

-Unify config schema:
  Standardize keys like paths.proc_root across all steps (right now you have a few naming variants like out_root vs intermediate_root).

- Smoke tests for entrypoints:
  Tiny tests that run each console script with --help and --print-config to catch argparse/config regressions.

- Add small tests for CLI + config precedence:
  Unit tests that assert CLI > YAML > defaults, and that missing credentials error messages are correct.

- Create e2e scripts that download, extract and processed

## Quality Checks:
- Rename path funciton in datasets/ prefix by get_* for options_chain & daily_features
- Make the top n violaitons as part of Spec for Soft

## Unit tests
- Add unit tests

## ORATS API Download & Extract
- Move constants like MAX_PER_CALL into config or constants at root

## ORATS Processed
- Make the build accept a Config objec tinstead of passign all args and maybe
find a way to use pydantic for this Config since we would let the user enter 
the data he wants

## Documentation
- Add module docstring for eahc file
- Add full docstrings for public API and shrot ones for internal helpers
- Add a doc for the ETL ORATS module like a graph with each step
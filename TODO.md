## Project
- Added CI wiht github/workflows
- Add Makefile for common dev actions like make test, make lint, make format, make test-integration ...
  and add a doc for dev worklow
- Add pyright to local + CI
- Added badges to README.md

## Scripts
- Create e2e scripts that download, extract and processed

## Quality Checks:
- Rename path funciton in datasets/ prefix by get_* for options_chain & daily_features
- Make the top n violaitons as part of Spec for Soft

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

## Project

## Release hygiene
Add CHANGELOG.md, LICENSE, CONTRIBUTING.md

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

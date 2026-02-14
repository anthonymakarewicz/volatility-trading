## Notebooks
- Finish the SPY ORATS validation notebook

## Source code
- Make part of the public API the plotting and helpers used across the notebooks.

## Quality Checks:
- Fix the thresholds for eahc checks (widen them)
- Fix the summarizer by bucket (DELTA bucket (-inf, 0])
- Rename path funciton in datasets/ prefix by get_* for options_chain & daily_features
- Make the top n violaitons as part of Spec for Soft

## Release hygiene
- Add CHANGELOG.md, CONTRIBUTING.md

## Scripts
- Create e2e scripts that download, extract and processed

## ORATS API Download & Extract
- Move constants like MAX_PER_CALL into config or constants at root

## ORATS Processed
- Make the build accept a Config objec tinstead of passign all args and maybe
find a way to use pydantic for this Config since we would let the user enter
the data he wants

## Source code
- Tighten the linter/formatter/type checkers and apply them for the wole src

## Project
- Improve the format on Github Pages, removign html and building a well presented webpage

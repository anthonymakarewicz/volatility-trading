## Project
- Finish setting up the jupytext worfklow
- Set up a Github Pages page to showcase the notebooks
- Don’t commit HTML, publish via GitHub Pages / Releases (Generate HTML in CI and publish)

## Notebooks
- Finish the SPY ORATS validation notebook

## Release hygiene
- Add CHANGELOG.md, CONTRIBUTING.md

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

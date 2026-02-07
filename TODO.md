## ORATS API Download & Extract
- Move constants like MAX_PER_CALL into config or constants at root

## ORATS Processed
- Make the build accept a Config objec tinstead of passign all args and maybe
find a way to use pydantic for this Config since we would let the user enter 
the data he wants

## Quality Checks:
- Refactor options QC into qc/options_chain

- Add info checks for daily_features (maybe first need to refatcor the info/suite.py
to make ROI run as optional; maybe ven remove ROI an dkepe optional)

- Refatcor the runners and suite runners to accept direclty the specs list

- Put shared code for options_chain & daily_features runners/helpers in shared/common

- Rename path funciton in datasets/ prefix by get_* for options_chain & daily_features

- Make the top n violaitons as part of Spec for Soft


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
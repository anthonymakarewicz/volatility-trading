# Troubleshooting

Use this page for repository-wide development and tooling issues.
For ORATS pipeline/app-specific failures, see
[ORATS Troubleshooting](reference/orats_troubleshooting.md).

## `pip-compile` Not Found

- Symptom: `zsh: command not found: pip-compile`
- Likely cause: `pip-tools` is not installed in the active virtual environment.
- Fix:
  - activate your venv (`source .venv/bin/activate`)
  - install dev dependencies (`pip install -r requirements-dev.txt`)
  - verify with `pip show pip-tools`
- Related docs:
  - [Development Guide](contributing/development.md)

## Wrong Python Version

- Symptom: dependency resolution mismatch vs CI, or missing wheels.
- Likely cause: local interpreter is not Python `3.12`.
- Fix:
  - check interpreter with `python --version`
  - recreate `.venv` with Python `3.12`
  - re-run dependency compile/install
- Related docs:
  - [Development Guide](contributing/development.md)

## Pre-commit Fails on Notebook Pair

- Symptom: Jupytext hook reports paired files inconsistent.
- Likely cause: `.ipynb` and `.py` are out of sync.
- Fix:
  - run `jupytext --sync notebooks/<name>.ipynb`
  - stage both files together (`git add notebooks/<name>.ipynb notebooks/<name>.py`)
- Related docs:
  - [Jupytext Workflow](contributing/jupytext.md)
  - [Notebook Authoring Guide](contributing/notebook_authoring.md)

## Pre-commit Reports "Git Index Is Outdated"

- Symptom: Jupytext says `git index is outdated` and asks to add paired file.
- Likely cause: hook updated one file, but the pair is not staged yet.
- Fix:
  - stage the updated paired file(s) (`git add notebooks/<name>.ipynb notebooks/<name>.py`)
  - run commit again
- Related docs:
  - [Jupytext Workflow](contributing/jupytext.md)

## Pytest Import or Discovery Errors

- Symptom: `ModuleNotFoundError` for `volatility_trading...` or import mismatch.
- Likely cause:
  - pytest config not aligned
  - duplicate test basenames across directories
- Fix:
  - ensure pytest config includes `testpaths = ["tests"]` and `pythonpath = ["src"]`
  - rename duplicate basenames like repeated `test_download.py`
- Related docs:
  - [Testing Guide](contributing/testing_guide.md)
  - [Test Authoring Guide](contributing/test_authoring.md)
  - [Tests README](../tests/README.md)

## CI Fails But Local Passes

- Symptom: local checks pass, CI fails on types/deps/hooks.
- Likely cause:
  - different Python version
  - stale compiled requirements
  - local checks narrower than CI
- Fix:
  - run `make ci` locally
  - recompile lock files using Python `3.12`
  - re-run `pre-commit run --all-files`
- Related docs:
  - [Development Guide](contributing/development.md)

# Jupytext Workflow

Use Jupytext to keep notebooks reviewable in Git while still runnable in Jupyter.

## Quickstart (Default)

1. Pair once per notebook:

```bash
jupytext --set-formats ipynb,py:percent notebooks/foo.ipynb
```

2. Edit either side:
- `notebooks/foo.ipynb` in Jupyter
- `notebooks/foo.py` in VS Code/text editor

3. Sync before commit:

```bash
make sync-nb NOTEBOOK=notebooks/foo.ipynb
```

You can pass either `.ipynb` or `.py`.

4. Commit both paired files (`.ipynb` + `.py`).

## If Sync Fails (Files Inconsistent)

If Jupytext reports that `.ipynb` and `.py` are inconsistent, pick one file as the
source of truth and regenerate the other:

Trust `.ipynb` and recreate `.py`:

```bash
jupytext --to py:percent notebooks/foo.ipynb
```

Trust `.py` and recreate `.ipynb`:

```bash
jupytext --to ipynb notebooks/foo.py
```

Then run:

```bash
jupytext --sync notebooks/foo.ipynb
```

## Optional Checks

Run from a clean kernel to catch hidden state:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/foo.ipynb
```

Clear outputs for cleaner diffs when desired:

```bash
jupyter nbconvert --clear-output --inplace notebooks/foo.ipynb
make sync-nb NOTEBOOK=notebooks/foo.ipynb
```

## Automation

Pre-commit includes a Jupytext sync hook for files under `notebooks/`.

Install once:

```bash
pre-commit install
```

Run manually when needed:

```bash
pre-commit run --all-files
```

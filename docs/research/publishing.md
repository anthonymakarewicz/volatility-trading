# Publishing Notebook Reports

This repository publishes notebook HTML reports via GitHub Pages using
GitHub Actions.

## Source of Truth

- Workflow file: `.github/workflows/pages.yml`
- Published site: `https://anthonymakarewicz.github.io/volatility-trading/`
- Notebook/report catalog: `docs/research/notebooks.md`

## How It Works

1. Push to `main` (or run the workflow manually).
2. GitHub Actions installs dependencies and runs `nbconvert`.
3. HTML files are generated into a temporary `site/` folder.
4. The workflow deploys `site/` to GitHub Pages.

Generated artifacts are not committed to the repo.

## Repo Policy

- Do not commit generated HTML report files.
- Keep local output folders ignored:
  - `reports/`
  - `site/`
- Keep notebook sources (`notebooks/*.ipynb`) in git.

## One-Time Setup In GitHub

In the repository settings:

`Settings -> Pages -> Source = GitHub Actions`

## Current Export Scope

The Pages workflow currently exports this set:

- `notebooks/greeks.ipynb`
- `notebooks/iv_surface_modelling.ipynb`
- `notebooks/rv_forecasting.ipynb`
- `notebooks/skew_trading.ipynb`
- `notebooks/vrp_harvesting.ipynb`

Update `.github/workflows/pages.yml` when adding/removing published notebooks.

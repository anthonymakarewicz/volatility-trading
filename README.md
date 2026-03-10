# **Volatility Trading on Equity Options**

[![CI](https://github.com/anthonymakarewicz/volatility-trading/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/anthonymakarewicz/volatility-trading/actions/workflows/ci.yml)
[![Pages](https://github.com/anthonymakarewicz/volatility-trading/actions/workflows/pages.yml/badge.svg?branch=main)](https://github.com/anthonymakarewicz/volatility-trading/actions/workflows/pages.yml)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-46a?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![Pyright](https://img.shields.io/badge/type%20check-pyright-blue)](https://github.com/microsoft/pyright)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/anthonymakarewicz/volatility-trading/blob/main/LICENSE)

This project develops and evaluates daily options-volatility strategies on index and single-stock underlyings.
Research spans the full pipeline: data engineering and quality checks, implied-volatility surface modelling, volatility forecasting, and strategy backtesting.
Backtests use realistic execution assumptions (bid/ask, slippage, commissions, position sizing, and risk limits) and are documented with reproducible notebooks and published reports.

Notebook reports (GitHub Pages): [https://anthonymakarewicz.github.io/volatility-trading/](https://anthonymakarewicz.github.io/volatility-trading/)

## **Quickstart**

1. Clone the repository:

```bash
git clone https://github.com/anthonymakarewicz/volatility-trading.git
cd volatility_trading
```

2. Install `uv` and create a virtual environment (Python 3.12+):

```bash
brew install uv  # or: pipx install uv
uv venv --python 3.12 .venv
source .venv/bin/activate
```

3. Install dependencies:

Primary contributor setup (editable package + dev tooling):

```bash
uv pip install -e ".[dev]"
```

Secondary options:
- Runtime-only install (users running package code without dev tools):

```bash
uv pip install .
```

- Editable runtime-only install (local source edits, no dev tools):

```bash
uv pip install -e .
```

`pip` remains supported as a fallback if you do not want to use `uv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

4. Optional: set credentials for ORATS data access:

```bash
cp .env.example .env
```

Then set `ORATS_API_KEY`, `ORATS_FTP_USER`, and `ORATS_FTP_PASS` in `.env`
if you plan to run the ORATS download/extract pipeline. You can skip this if
you are using already-prepared data or a different options data source.

## **ORATS ETL Pipeline (End-to-End)**

Pipeline steps:
- API download
- API extract
- FTP download
- FTP extract
- Build options chain
- Build daily features
- QC options chain
- QC daily features

Use `--dry-run` to validate config, paths, and credentials before running writes/network.

```bash
orats-api-download --config config/orats/api_download.yml --dry-run
```

For the full command sequence, see [Data pipeline](https://github.com/anthonymakarewicz/volatility-trading/blob/main/docs/reference/data_pipeline.md).

## **Current Data Support Status**

- Options ETL (options chain + daily features) is currently supported through the ORATS pipeline.
- External feed sync (`fred-sync`, `yfinance-sync`) currently covers rates/market time series, not a full generic options ETL path.
- The options backtesting runtime expects the current project options-chain schema (for example quotes/Greeks fields used by entry, sizing, and lifecycle).
- You can run backtests with non-ORATS data if it is pre-normalized to the expected schema.

## **Stability**

- Current release line is `0.4.x` (**alpha / pre-1.0**).
- Public APIs, data contracts, and configuration surfaces may evolve between minor versions.
- For reproducible research, pin exact package versions and review [CHANGELOG.md](https://github.com/anthonymakarewicz/volatility-trading/blob/main/CHANGELOG.md) before upgrading.
- Public vs internal boundaries are defined in [API Scope](https://github.com/anthonymakarewicz/volatility-trading/blob/main/docs/reference/api_scope.md).

## **Data Contract / Supported Inputs**

| Source | Input expected by backtester | Support status | Adapter path |
|:--|:--|:--|:--|
| ORATS ETL output | Canonical long-format options chain | First-class | `CanonicalOptionsChainAdapter` (or mode `canonical`) |
| OptionsDX ETL output | Cleaned long-format panel (`reshape='long'`) | Supported | `OptionsDxOptionsChainAdapter` |
| Custom/vendor dataset | Long-format panel mapped to canonical fields | Supported with mapping | `ColumnMapOptionsChainAdapter` |

Notes:
- Raw wide OptionsDX vendor format (`c_*` / `p_*`) is not accepted directly by the backtester.
- Contract details and adapter behavior: [Options Data Adapters](https://github.com/anthonymakarewicz/volatility-trading/blob/main/docs/reference/options_data_adapters.md)
- ORATS pipeline reference: [Data Pipeline](https://github.com/anthonymakarewicz/volatility-trading/blob/main/docs/reference/data_pipeline.md)
- OptionsDX onboarding: [OptionsDX Setup](https://github.com/anthonymakarewicz/volatility-trading/blob/main/docs/reference/optionsdx_setup.md)

## **Quick VRP Backtest Example**

Assume you already prepared:

- `options`: long-format `pandas` options panel indexed by `trade_date`

1. Import strategy and backtesting types:

```python
from volatility_trading.backtesting import (
    AccountConfig,
    Backtester,
    BidAskFeeOptionExecutionModel,
    BacktestRunConfig,
    BrokerConfig,
    FixedBpsHedgeExecutionModel,
    ExecutionConfig,
    HedgeMarketData,
    MarginConfig,
    OptionsBacktestDataBundle,
    OptionsMarketData,
    print_performance_report,
    to_daily_mtm,
)
from volatility_trading.options import RegTMarginModel
from volatility_trading.signals import ShortOnlySignal
from volatility_trading.strategies import VRPHarvestingSpec, make_vrp_strategy
```

2. Build the backtest data bundle:

```python
hedge_mid = options.groupby(level=0)["spot_price"].first().astype(float)
data = OptionsBacktestDataBundle(
    options_market=OptionsMarketData(
        chain=options,
    ),
    features=None,
    hedge_market=HedgeMarketData(mid=hedge_mid),
)
```

3. Define strategy spec and run configuration:

```python
vrp_spec = VRPHarvestingSpec(
    signal=ShortOnlySignal(),
    rebalance_period=10,
    risk_budget_pct=0.03,
    margin_budget_pct=0.4,
)
strategy = make_vrp_strategy(vrp_spec)

cfg = BacktestRunConfig(
    account=AccountConfig(initial_capital=50_000),
    execution=ExecutionConfig(
        option_execution_model=BidAskFeeOptionExecutionModel(
            commission_per_leg=0.0,
        ),
        hedge_execution_model=FixedBpsHedgeExecutionModel(
            fee_bps=0.0,
        ),
    ),
    broker=BrokerConfig(
        margin=MarginConfig(model=RegTMarginModel(broad_index=False))
    ),
)
```

4. Run the backtest and compute daily MTM/performance metrics:

```python
bt = Backtester(
    data=data,
    strategy=strategy,
    config=cfg,
)
trades, mtm = bt.run()
daily_mtm = to_daily_mtm(mtm, cfg.account.initial_capital)

metrics = print_performance_report(
    trades=trades,
    mtm_daily=daily_mtm,
    risk_free_rate=0.02,
)
```

For a full scriptable workflow (data loading + backtest run), see
[VRP end-to-end example](https://github.com/anthonymakarewicz/volatility-trading/blob/main/examples/backtesting/vrp_end_to_end.py).
For focused backtesting configuration examples (execution, margin, adapters, hedging), see
[examples/README.md](https://github.com/anthonymakarewicz/volatility-trading/blob/main/examples/README.md).
For hedging model semantics and WW/fixed-band configuration details, see
[hedging.md](https://github.com/anthonymakarewicz/volatility-trading/blob/main/docs/reference/backtesting/hedging.md).
For option execution model behavior and option-cost attribution fields, see
[option_execution.md](https://github.com/anthonymakarewicz/volatility-trading/blob/main/docs/reference/backtesting/option_execution.md).
For the research-style workflow and reporting exploration, see
[VRP notebook](https://github.com/anthonymakarewicz/volatility-trading/blob/main/notebooks/vrp_harvesting/notebook.py).

Preferred imports for common backtesting usage come from
`volatility_trading.backtesting`. The narrower
`volatility_trading.backtesting.options_engine` namespace remains available for
advanced engine-specific helpers.

## **Advanced Option Execution Injection**

`Backtester` intentionally keeps a stable high-level API and does not expose an
`option_execution_model` argument.

If you want to override option execution behavior, set it on
`BacktestRunConfig.execution.option_execution_model`:

```python
from volatility_trading.backtesting import (
    BacktestRunConfig,
    Backtester,
    ExecutionConfig,
    MidNoCostOptionExecutionModel,
)

cfg = BacktestRunConfig(
    execution=ExecutionConfig(
        option_execution_model=MidNoCostOptionExecutionModel(),
    ),
)
bt = Backtester(
    data=data,
    strategy=strategy,
    config=cfg,
)
trades, mtm = bt.run()
```

## **Tests**

Run unit tests (default):

```bash
pytest -q
```

Run integration tests:

```bash
pytest -q -m integration
```

See [Testing guide](https://github.com/anthonymakarewicz/volatility-trading/blob/main/docs/contributing/testing_guide.md) for layout and conventions.

## **Continuous Integration (CI)**

GitHub Actions runs:
- Ruff lint + format checks
- Pyright type checks
- Unit tests by default
- Integration tests on PRs and pushes to `main` (and manual runs)

See [CI workflow](https://github.com/anthonymakarewicz/volatility-trading/blob/main/.github/workflows/ci.yml).

## **Developer Workflow**

Common commands are available via `Makefile`:

```bash
make lint
make format
make check
make typecheck
make test
make test-unit
make test-integration
make package-check
make sync-nb
make sync-nb-all
make ci
```

For full setup and tooling details, see the
[Documentation index](https://github.com/anthonymakarewicz/volatility-trading/blob/main/docs/README.md).
Notebook HTML reports are built in GitHub Actions and published to GitHub Pages.

## **Docs**

See [Documentation index](https://github.com/anthonymakarewicz/volatility-trading/blob/main/docs/README.md) for the full docs map.
Most-used pages:
- [Development guide](https://github.com/anthonymakarewicz/volatility-trading/blob/main/docs/contributing/development.md)
- [Testing guide](https://github.com/anthonymakarewicz/volatility-trading/blob/main/docs/contributing/testing_guide.md)
- [Notebook catalog](https://github.com/anthonymakarewicz/volatility-trading/blob/main/docs/research/notebooks.md)


## **Research Highlights**

We publish research notebooks and strategy diagnostics as HTML reports on GitHub Pages.

- RV forecasting: HAR-RV-VIX reaches about **30% OOS R²** vs a naive baseline.
- IV surface modelling: parametric vs non-parametric surface comparison workflows.
- Skew trading: delta-hedged RR strategy with realistic costs and risk controls.

Explore details here:

- [Research results (detailed)](https://github.com/anthonymakarewicz/volatility-trading/blob/main/docs/research/results.md)
- [Notebook catalog](https://github.com/anthonymakarewicz/volatility-trading/blob/main/docs/research/notebooks.md)
- [Published HTML reports](https://anthonymakarewicz.github.io/volatility-trading/)

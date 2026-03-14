# Examples

Use these modules as runnable references for options backtesting setup.

## Run

From repository root, run examples with `python -m ...`:

```bash
python -m examples.backtesting.minimal_vrp_backtest
python -m examples.backtesting.strategies.vrp_end_to_end
python -m examples.backtesting.strategies.skew_mispricing_end_to_end
python -m examples.backtesting.execution.models_and_costs
python -m examples.backtesting.margin.model_and_financing
python -m examples.backtesting.margin.policy_accounts
python -m examples.backtesting.hedging.configuration
python -m examples.backtesting.adapters.options_market
python -m examples.backtesting.hedging.fixed_band
python -m examples.backtesting.hedging.ww_band
python -m examples.backtesting.hedging.cost_baselines
```

## Shared Data Prep Helpers

The examples use the public backtesting helper layer for common market-data
prep tasks:

- `load_orats_options_chain_for_backtest(...)`
- `load_fred_rate_series(...)`
- `load_yfinance_close_series(...)`
- `spot_series_from_options_chain(...)`

`load_orats_options_chain_for_backtest(...)` also supports common notebook
filters:

- `start`
- `end`
- `dte_min`
- `dte_max`

Example-level wrappers in `examples/core/backtesting_helpers.py` are kept only
when they add composition behavior such as window slicing, rate alignment, or
full backtester construction.

## Minimal / Strategy End-to-End

- `examples/backtesting/minimal_vrp_backtest.py`
  - smallest public-API VRP example showing data bundle, strategy, run config, and reporting
- `examples/backtesting/strategies/vrp_end_to_end.py`
  - full VRP workflow (load data, build strategy/config, run, report)
- `examples/backtesting/strategies/skew_mispricing_end_to_end.py`
  - full skew workflow (load options + daily features, build strategy, run, report)

## Focused Execution Example

- `examples/backtesting/execution/models_and_costs.py`
  - compare realistic option/hedge execution against no-cost baselines

## Focused Margin Examples

- `examples/backtesting/margin/model_and_financing.py`
  - compare margin-model and financing setups using a rate series
- `examples/backtesting/margin/policy_accounts.py`
  - compare account-level liquidation and financing policies

## Focused Hedging Examples

- `examples/backtesting/hedging/configuration.py`
  - compare disabled, fixed-band, and WW-style hedging policies
- `examples/backtesting/hedging/fixed_band.py`
  - fixed delta no-trade band configuration
- `examples/backtesting/hedging/ww_band.py`
  - Whalley-Wilmott-style dynamic band configuration
- `examples/backtesting/hedging/cost_baselines.py`
  - compare fixed-bps costs vs zero-cost baseline assumptions

## Focused Adapter Example

- `examples/backtesting/adapters/options_market.py`
  - show how `OptionsMarketData` scopes per-dataset adapter configuration

## Common CLI Args

Most parameterized scripts support:

- `--ticker`
- `--start`
- `--end`
- `--initial-capital`
- `--commission-per-leg`
- `--hedge-fee-bps`

The full end-to-end script also supports:

- `--rebalance-period`
- `--risk-budget-pct`
- `--margin-budget-pct`

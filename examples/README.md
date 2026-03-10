# Examples

Use these modules as runnable references for options backtesting setup.

## Run

From repository root, run examples with `python -m ...`:

```bash
python -m examples.backtesting.minimal_options_backtest
python -m examples.backtesting.vrp_end_to_end
python -m examples.backtesting.execution.models_and_costs
python -m examples.backtesting.margin.model_and_financing
python -m examples.backtesting.margin.policy_accounts
python -m examples.backtesting.hedging.configuration
python -m examples.backtesting.adapters.options_market
python -m examples.backtesting.hedging.fixed_band
python -m examples.backtesting.hedging.ww_band
python -m examples.backtesting.hedging.cost_baselines
```

## Minimal / End-to-End

- `examples/backtesting/minimal_options_backtest.py`
  - smallest public-API example showing data bundle, strategy, run config, and reporting
- `examples/backtesting/vrp_end_to_end.py`
  - full VRP workflow (load data, build strategy/config, run, report)

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

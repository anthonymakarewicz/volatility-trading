# Options Engine Option Execution

This page documents option-leg execution assumptions in the options backtesting
engine: fill-price models, explicit trade-cost attribution, and where custom
execution models can be injected.

## Default Behavior

`ExecutionConfig` resolves an option execution model at config construction.

Default resolution behavior:

- if `execution.option_execution_model` is provided, that model is used
- otherwise config builds `BidAskFeeOptionExecutionModel` from:
  - `ExecutionConfig.slip_ask`
  - `ExecutionConfig.slip_bid`
  - `ExecutionConfig.commission_per_leg`

That model applies:

- side-aware reference price (`ask` for buys, `bid` for sells, fallback `mid`)
- side-aware slippage from `ExecutionConfig.slip_ask` / `ExecutionConfig.slip_bid`
- commission from `ExecutionConfig.commission_per_leg`

The engine also exposes `MidNoCostOptionExecutionModel` for baseline runs.

## Accounting Attribution

Option market movement and option transaction costs are separated:

- MTM fields:
  - `MtmRecord.option_market_pnl`
  - `MtmRecord.option_trade_cost`
- Trade fields:
  - `TradeRecord.option_entry_cost`
  - `TradeRecord.option_exit_cost`
- Per-leg context:
  - `TradeLegRecord.entry_mid_price`
  - `TradeLegRecord.exit_mid_price`

This keeps market-PnL attribution distinct from execution friction.

## Injection Boundary

`Backtester` does not expose an `option_execution_model` argument.

Custom option execution is injected at the plan-builder boundary:

- `BacktestRunConfig.execution.option_execution_model=...`

```python
from volatility_trading.backtesting import BacktestRunConfig, ExecutionConfig
from volatility_trading.backtesting.engine import Backtester
from volatility_trading.backtesting.options_engine import MidNoCostOptionExecutionModel

run_config = BacktestRunConfig(
    execution=ExecutionConfig(
        lot_size=100,
        option_execution_model=MidNoCostOptionExecutionModel(),
    ),
)

bt = Backtester(
    data=data_bundle,
    strategy=strategy,
    config=run_config,
)
trades, mtm = bt.run()
```

## Notes

- Entry and exit option costs are charged on their actual lifecycle dates (not
  as an upfront roundtrip lump).
- If same-day reentry is enabled, MTM can include entry costs for a newly opened
  unresolved position while `trades` only includes closed positions.

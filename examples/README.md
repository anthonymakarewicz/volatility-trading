# Examples

Use these scripts as executable references for options backtesting setup.

## End-to-End

- `examples/vrp_end_to_end.py`
  - full VRP workflow (load data, build strategy/config, run, report)

## Focused Delta-Hedging Examples

- `examples/vrp_hedging_fixed_band.py`
  - fixed delta no-trade band configuration
- `examples/vrp_hedging_ww_band.py`
  - Whalley-Wilmott-style dynamic band configuration
- `examples/vrp_hedging_cost_baselines.py`
  - compare fixed-bps costs vs zero-cost baseline assumptions

## Run

From repository root:

```bash
python examples/vrp_end_to_end.py
python examples/vrp_hedging_fixed_band.py
python examples/vrp_hedging_ww_band.py
python examples/vrp_hedging_cost_baselines.py
```

## Common CLI Args

All scripts support:

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

## More Details

- WW/fixed-band hedging guide:
  - `docs/reference/backtesting/hedging.md`

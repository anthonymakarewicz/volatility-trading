# Options Engine Hedging Guide

This page describes how to configure dynamic delta hedging in the options
backtesting engine, with emphasis on fixed bands vs Whalley-Wilmott (WW)-style
dynamic bands.

## Configuration Surface

Delta hedging is configured in strategy lifecycle policy:

- `DeltaHedgePolicy`
- `HedgeTriggerPolicy`
- `FixedDeltaBandModel` or `WWDeltaBandModel`

Execution cost assumptions are run-level settings:

- `ExecutionConfig.hedge_execution_model`

## Fixed Band (Static)

Use a constant absolute no-trade half-width around `target_net_delta`.

```python
from volatility_trading.backtesting.options_engine import (
    DeltaHedgePolicy,
    FixedDeltaBandModel,
    HedgeTriggerPolicy,
)

delta_hedge = DeltaHedgePolicy(
    enabled=True,
    target_net_delta=0.0,
    trigger=HedgeTriggerPolicy(
        band_model=FixedDeltaBandModel(half_width_abs=25.0),
        rebalance_every_n_days=5,
        combine_mode="or",
    ),
    rebalance_to="center",
    min_rebalance_qty=1.0,
)
```

Use this when you want simple and stable hedge behavior.

## WW Band (Dynamic)

WW computes a dynamic no-trade band from market/greek context and transaction
cost assumptions.

Runtime form:

`band = c * (fee_rate / (gamma_eff * sigma_eff * spot_eff^2))^(1/3)`

with floors/clamps:

- `gamma_eff = max(abs(gamma), gamma_floor)`
- `sigma_eff = max(volatility, sigma_floor)`
- `spot_eff = max(spot, spot_floor)`
- final `band` clamped to `[min_band_abs, max_band_abs]`

Fee source:

- `fee_bps_override` from `WWDeltaBandModel`, else
- `ExecutionConfig.hedge_execution_model.fee_bps` when available
- falls back to `0.0` if the configured hedge model does not expose `fee_bps`

```python
from volatility_trading.backtesting.options_engine import (
    DeltaHedgePolicy,
    HedgeTriggerPolicy,
    WWDeltaBandModel,
)

delta_hedge = DeltaHedgePolicy(
    enabled=True,
    target_net_delta=0.0,
    trigger=HedgeTriggerPolicy(
        band_model=WWDeltaBandModel(
            calibration_c=1.0,
            min_band_abs=5.0,
            max_band_abs=40.0,
        ),
        rebalance_every_n_days=None,
        combine_mode="or",
    ),
    rebalance_to="nearest_boundary",
    min_rebalance_qty=1.0,
)
```

Important:

- `WWDeltaBandModel` requires `rebalance_to="nearest_boundary"` by validation.

## Rebalance Semantics

- `rebalance_to="center"`:
  - when triggered, target is always `target_net_delta`.
- `rebalance_to="nearest_boundary"`:
  - outside band -> rebalance to crossed boundary,
  - inside band -> no recenter trade.

## Trigger Combination

`HedgeTriggerPolicy` supports:

- band trigger (`band_model`)
- periodic trigger (`rebalance_every_n_days`)

combined with:

- `combine_mode="or"`: either trigger is enough
- `combine_mode="and"`: both triggers required

## Execution Cost Models

Default hedge execution model is fixed-bps with side-aware reference price:

- buy uses ask when available,
- sell uses bid when available,
- slippage from `slip_ask` / `slip_bid`,
- bps fee from `fee_bps`.

For a baseline/debug run, use `MidNoCostExecutionModel` (fill at mid, no
explicit trade cost).

## Related Examples

- [`examples/vrp_end_to_end.py`](../../examples/vrp_end_to_end.py)
- [`examples/vrp_hedging_fixed_band.py`](../../examples/vrp_hedging_fixed_band.py)
- [`examples/vrp_hedging_ww_band.py`](../../examples/vrp_hedging_ww_band.py)
- [`examples/vrp_hedging_cost_baselines.py`](../../examples/vrp_hedging_cost_baselines.py)

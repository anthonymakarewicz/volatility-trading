# Options Data Adapters

The options backtesting runtime now supports a typed adapter boundary before
plan compilation:

1. `normalize` source columns to canonical engine names
2. `validate` required schema/dtypes
3. `run` strategy plan compilation and lifecycle

This boundary is applied in `build_options_execution_plan(...)`.

Canonical field names are centralized in
`volatility_trading.contracts.options_chain` and reused by ETL and
backtesting adapters.

## Canonical Options-Chain Contract

Required columns:
- `trade_date` (or DatetimeIndex)
- `expiry_date`
- `dte`
- `option_type` (`C`/`P`, case-insensitive `call`/`put` accepted)
- `strike`
- `delta`
- `bid_price`
- `ask_price`

Optional columns:
- `gamma`
- `vega`
- `theta`
- `spot_price`
- `market_iv`
- `model_iv`
- `yte`
- `open_interest`
- `volume`

## Built-in Adapters

- `OratsOptionsChainAdapter`
  - Default adapter when `options_adapter=None`.
  - Handles common ORATS-style aliases (for example `date`, `expiry`, `bid`, `ask`).

- `YfinanceOptionsChainAdapter`
  - Best-effort yfinance normalization.
  - Parses `option_type` from `contract_symbol` when needed.
  - Still requires `delta` to be present for delta-target leg selection.

- `OptionsDxOptionsChainAdapter`
  - Supports cleaned OptionsDX long-format chains.
  - Maps vendor `iv` to canonical `market_iv`.
  - Rejects raw wide vendor format (`c_*`/`p_*`) with a clear error.

- `ColumnMapOptionsChainAdapter`
  - Generic source-to-canonical mapping for custom datasets.

- `CanonicalOptionsChainAdapter`
  - Fast-path contract check for trusted canonical ETL outputs.
  - Skips alias remapping/coercion and enforces schema validation only.
  - Requires canonical typed columns (for example numeric `delta`, datetime `expiry_date`).

## Usage

```python
from volatility_trading.backtesting import OptionsBacktestDataBundle
from volatility_trading.backtesting.options_engine import ColumnMapOptionsChainAdapter

adapter = ColumnMapOptionsChainAdapter(
    source_to_canonical={
        "qdt": "trade_date",
        "exp": "expiry_date",
        "days": "dte",
        "cp": "option_type",
        "k": "strike",
        "d": "delta",
        "b": "bid_price",
        "a": "ask_price",
    }
)

data = OptionsBacktestDataBundle(
    options=raw_options_df,
    options_adapter=adapter,
)
```

## Validation Errors

Schema issues raise `OptionsChainAdapterError` with explicit messages, for example:
- missing required canonical columns
- unparseable `trade_date`
- invalid `option_type` labels
- required numeric columns becoming all-null after coercion

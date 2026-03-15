# Options Data Adapters

The options backtesting runtime now supports a typed adapter boundary before
runtime input construction:

1. `normalize` source columns to canonical engine names
2. `validate` the canonical options-chain contract
3. `construct` canonical runtime inputs such as `OptionsMarketData(...)`

Canonical field names are centralized in
`volatility_trading.contracts.options_chain` and reused by ETL and
backtesting adapters.
Provider-specific alias/rename mappings live in
`volatility_trading.config.options_chain_sources`.

`OptionsMarketData` is canonical-only:
- pass a canonical long options panel into `OptionsMarketData(chain=...)`
- use adapters through explicit canonicalization helpers before runtime

Preferred user-facing entrypoints:
- `canonicalize_options_chain_for_backtest(...)`
- `load_orats_options_chain_for_backtest(...)`

Both return canonical long `pandas.DataFrame` data indexed by `trade_date`.

## Choose An Ingestion Path

Use these ingestion paths based on where the options data comes from and how
much normalization work is still needed.

- processed-source loader
  - use `load_orats_options_chain_for_backtest(...)`
  - this is for the repo's processed ORATS parquet outputs
  - it applies supported source filters before reshaping and then validates the
    result through the canonical strict contract

- vendor adapter
  - use `OratsOptionsChainAdapter`, `OptionsDxOptionsChainAdapter`, or
    `YfinanceOptionsChainAdapter`
  - this is for users who already have a raw vendor-style dataframe in memory
    and want to normalize it into the canonical backtesting contract

- generic custom-schema adapter
  - use `ColumnMapOptionsChainAdapter`
  - this is for custom dataframes whose column names do not match one of the
    built-in vendor adapters

- canonical strict adapter
  - use `CanonicalOptionsChainAdapter`
  - this is for already-canonical long options data where you want strict
    contract validation without alias remapping or coercive normalization

Processed loaders still validate the returned chain. They bypass coercive
vendor normalization for trusted internal processed data; they do not bypass
contract validation.

## Canonical Options-Chain Contract

Canonical loaders and adapters return a long `pandas.DataFrame` whose index is
`trade_date`.

Required columns:
- `trade_date` (or DatetimeIndex during input validation)
- `expiry_date`
- `dte`
- `option_type` (`C`/`P` in the canonical contract)
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

Notes:
- strict canonical inputs should already use `option_type` values `C` / `P`
- coercive vendor-adapter paths may accept common labels such as `call` / `put`
  and normalize them into the canonical contract

## Built-in Adapters

- `OratsOptionsChainAdapter`
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

## Polars Boundary

Adapter-based ingestion accepts either:
- `pandas.DataFrame`
- `polars.DataFrame`

Polars input is converted once at the adapter boundary. Canonical returned
chains and core backtesting runtime inputs remain pandas-based.

## Examples

Processed ORATS convenience loader:

```python
from volatility_trading.backtesting import load_orats_options_chain_for_backtest

options = load_orats_options_chain_for_backtest(
    "SPY",
    start="2011-01-01",
    end="2017-12-31",
    dte_min=5,
    dte_max=60,
)
```

Custom dataset via `ColumnMapOptionsChainAdapter`:

```python
from volatility_trading.backtesting import (
    ColumnMapOptionsChainAdapter,
    OptionsBacktestDataBundle,
    OptionsMarketData,
    canonicalize_options_chain_for_backtest,
)

adapter = ColumnMapOptionsChainAdapter(
    source_to_canonical={
        "your_trade_date_col": "trade_date",
        "your_expiry_col": "expiry_date",
        "your_dte_col": "dte",
        "your_type_col": "option_type",
        "your_strike_col": "strike",
        "your_delta_col": "delta",
        "your_bid_col": "bid_price",
        "your_ask_col": "ask_price",
    }
)

options = canonicalize_options_chain_for_backtest(
    raw_options_df,
    adapter=adapter,
)

data = OptionsBacktestDataBundle(
    options_market=OptionsMarketData(chain=options),
)
```

Raw ORATS-style dataframe via `OratsOptionsChainAdapter`:

```python
from volatility_trading.backtesting import (
    OratsOptionsChainAdapter,
    OptionsBacktestDataBundle,
    OptionsMarketData,
    canonicalize_options_chain_for_backtest,
)

adapter = OratsOptionsChainAdapter()

options = canonicalize_options_chain_for_backtest(
    raw_orats_df,
    adapter=adapter,
)

data = OptionsBacktestDataBundle(
    options_market=OptionsMarketData(chain=options),
)
```

Already-canonical long dataframe via `CanonicalOptionsChainAdapter`:

```python
from volatility_trading.backtesting import (
    CanonicalOptionsChainAdapter,
    OptionsBacktestDataBundle,
    OptionsMarketData,
    canonicalize_options_chain_for_backtest,
)

options = canonicalize_options_chain_for_backtest(
    canonical_options_df,
    adapter=CanonicalOptionsChainAdapter(),
)

data = OptionsBacktestDataBundle(
    options_market=OptionsMarketData(chain=options),
)
```

## Validation Errors

Schema issues raise `OptionsChainAdapterError` with explicit messages, for example:
- missing required canonical columns
- unparseable `trade_date`
- invalid `option_type` labels
- required numeric columns becoming all-null after coercion

## Related Example

- [`examples/backtesting/adapters/options_market.py`](../../examples/backtesting/adapters/options_market.py)

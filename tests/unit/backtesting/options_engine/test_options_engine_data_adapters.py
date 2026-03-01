from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import pandas as pd
import pytest

from volatility_trading.backtesting import (
    AccountConfig,
    BacktestRunConfig,
    ExecutionConfig,
    OptionsBacktestDataBundle,
)
from volatility_trading.backtesting.engine import Backtester
from volatility_trading.backtesting.options_engine import (
    CanonicalOptionsChainAdapter,
    ColumnMapOptionsChainAdapter,
    LegSpec,
    LifecycleConfig,
    OptionsChainAdapterError,
    OptionsDxOptionsChainAdapter,
    OratsOptionsChainAdapter,
    StrategySpec,
    StructureSpec,
    YfinanceOptionsChainAdapter,
    normalize_options_chain,
    validate_options_chain,
)
from volatility_trading.options import OptionType
from volatility_trading.signals.base_signal import Signal


class _AlwaysShortSignal(Signal):
    def generate_signals(self, data: pd.Series | pd.DataFrame) -> pd.DataFrame:
        idx = data.index
        return pd.DataFrame({"long": False, "short": True}, index=idx)

    def get_params(self) -> dict:
        return {}

    def set_params(self, **kwargs):
        _ = kwargs


class _AdapterLike(Protocol):
    @property
    def name(self) -> str: ...

    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame: ...


def _build_orats_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01"],
            "expiry": ["2020-01-31", "2020-01-31"],
            "dte": [" 30 ", " 30 "],
            "option_type": ["call", "put"],
            "strike": [" 100.0", "100.0 "],
            "delta": [" 0.50", "-0.50 "],
            "bid": [" 4.9", "5.1"],
            "ask": [" 5.2", "5.4 "],
        }
    )


def _build_optionsdx_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "QUOTE_DATE": ["2020-01-02", "2020-01-02"],
            "EXPIRE_DATE": ["2020-01-31", "2020-01-31"],
            "DTE": [29, 29],
            "OPTION_TYPE": ["C", "P"],
            "STRIKE": [325.0, 325.0],
            "DELTA": [0.42, -0.58],
            "GAMMA": [0.01, 0.01],
            "VEGA": [0.10, 0.10],
            "THETA": [-0.05, -0.05],
            "BID": [4.8, 5.0],
            "ASK": [5.0, 5.3],
            "IV": [0.18, 0.19],
            "UNDERLYING_LAST": [326.0, 326.0],
            "VOLUME": [100.0, 90.0],
        }
    )


def _build_column_map_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "qdt": ["2020-01-01"],
            "exp": ["2020-01-31"],
            "days": [30],
            "cp": ["C"],
            "k": [100.0],
            "d": [0.5],
            "b": [5.0],
            "a": [5.2],
        }
    )


def _column_map_adapter() -> ColumnMapOptionsChainAdapter:
    return ColumnMapOptionsChainAdapter(
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


@pytest.mark.parametrize(
    ("raw_factory", "adapter_factory", "expects_market_iv"),
    [
        (_build_orats_raw, OratsOptionsChainAdapter, False),
        (_build_optionsdx_raw, OptionsDxOptionsChainAdapter, True),
        (_build_column_map_raw, _column_map_adapter, False),
    ],
    ids=["orats", "optionsdx", "column_map"],
)
def test_adapter_matrix_normalizes_to_canonical(
    raw_factory: Callable[[], pd.DataFrame],
    adapter_factory: Callable[[], _AdapterLike],
    expects_market_iv: bool,
):
    normalized = normalize_options_chain(raw_factory(), adapter=adapter_factory())

    assert normalized.index.name == "trade_date"
    assert isinstance(normalized.index, pd.DatetimeIndex)
    assert set(normalized["option_type"].unique()) <= {"C", "P"}
    assert "bid_price" in normalized.columns
    assert "ask_price" in normalized.columns
    assert normalized["delta"].dtype.kind == "f"
    assert ("market_iv" in normalized.columns) is expects_market_iv


def test_canonical_adapter_validates_trusted_canonical_input():
    raw = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"]),
            "expiry_date": pd.to_datetime(["2020-01-31"]),
            "dte": [30.0],
            "option_type": ["C"],
            "strike": [100.0],
            "delta": [0.5],
            "bid_price": [5.0],
            "ask_price": [5.2],
        }
    )
    normalized = normalize_options_chain(raw, adapter=CanonicalOptionsChainAdapter())
    assert normalized.index.name == "trade_date"
    assert normalized["option_type"].iloc[0] == "C"


def test_validate_options_chain_supports_coerce_and_strict_modes():
    raw = pd.DataFrame(
        {
            "trade_date": ["2020-01-01"],
            "expiry_date": ["2020-01-31"],
            "dte": ["30"],
            "option_type": ["call"],
            "strike": ["100.0"],
            "delta": ["0.5"],
            "bid_price": ["5.0"],
            "ask_price": ["5.2"],
        }
    )
    coerced = validate_options_chain(raw, adapter_name="test", validation_mode="coerce")
    assert coerced["delta"].dtype.kind == "f"
    assert coerced["option_type"].iloc[0] == "C"

    with pytest.raises(
        OptionsChainAdapterError,
        match="must be datetime-like for strict validation",
    ):
        validate_options_chain(raw, adapter_name="test", validation_mode="strict")


def test_canonical_adapter_raises_on_non_numeric_required_column():
    raw = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"]),
            "expiry_date": pd.to_datetime(["2020-01-31"]),
            "dte": [30.0],
            "option_type": ["C"],
            "strike": [100.0],
            "delta": ["bad_value"],
            "bid_price": [5.0],
            "ask_price": [5.2],
        }
    )
    with pytest.raises(
        OptionsChainAdapterError,
        match="required numeric column 'delta' must be numeric",
    ):
        normalize_options_chain(raw, adapter=CanonicalOptionsChainAdapter())


def test_yfinance_adapter_parses_option_type_from_contract_symbol():
    raw = pd.DataFrame(
        {
            "quote_date": ["2024-01-02", "2024-01-02"],
            "expiration": ["2024-01-19", "2024-01-19"],
            "dte": [17, 17],
            "contract_symbol": ["SPY240119C00450000", "SPY240119P00450000"],
            "strike": [450.0, 450.0],
            "delta": [0.45, -0.55],
            "bid": [4.8, 5.1],
            "ask": [5.0, 5.4],
        }
    )
    normalized = normalize_options_chain(raw, adapter=YfinanceOptionsChainAdapter())
    assert set(normalized["option_type"].unique()) == {"C", "P"}


def test_optionsdx_adapter_raises_on_wide_vendor_input():
    raw = pd.DataFrame(
        {
            "QUOTE_DATE": ["2020-01-02"],
            "EXPIRE_DATE": ["2020-01-31"],
            "DTE": [29],
            "STRIKE": [325.0],
            "C_DELTA": [0.42],
            "C_BID": [4.8],
            "C_ASK": [5.0],
            "P_DELTA": [-0.58],
            "P_BID": [5.0],
            "P_ASK": [5.3],
        }
    )
    with pytest.raises(OptionsChainAdapterError, match="wide c_\\*/p_\\* columns"):
        normalize_options_chain(raw, adapter=OptionsDxOptionsChainAdapter())


def test_adapter_validation_raises_on_missing_required_columns():
    raw = pd.DataFrame(
        {
            "trade_date": ["2020-01-01"],
            "expiry_date": ["2020-01-31"],
            "dte": [30],
            "option_type": ["C"],
            "strike": [100.0],
            "bid_price": [5.0],
            "ask_price": [5.2],
        }
    )
    with pytest.raises(
        OptionsChainAdapterError, match="missing required canonical columns"
    ):
        normalize_options_chain(raw, adapter=OratsOptionsChainAdapter())


def test_adapter_validation_raises_when_required_numeric_is_all_null():
    raw = pd.DataFrame(
        {
            "trade_date": ["2020-01-01"],
            "expiry_date": ["2020-01-31"],
            "dte": [30],
            "option_type": ["C"],
            "strike": [100.0],
            "delta": ["bad_value"],
            "bid_price": [5.0],
            "ask_price": [5.2],
        }
    )
    with pytest.raises(
        OptionsChainAdapterError, match="required numeric column 'delta' is all-null"
    ):
        normalize_options_chain(raw, adapter=OratsOptionsChainAdapter())


def test_polars_dataframe_supported_at_adapter_boundary():
    pl = pytest.importorskip("polars")
    raw = pl.DataFrame(_build_orats_raw())
    normalized = normalize_options_chain(raw, adapter=OratsOptionsChainAdapter())
    assert isinstance(normalized, pd.DataFrame)
    assert normalized.index.name == "trade_date"


def test_backtester_uses_data_bundle_options_adapter_boundary():
    options_raw = pd.DataFrame(
        [
            {
                "qdt": "2020-01-01",
                "exp": "2020-01-31",
                "days": 30,
                "cp": "C",
                "k": 100.0,
                "d": 0.5,
                "b": 5.0,
                "a": 5.2,
                "spot": 100.0,
                "ivm": 0.20,
            },
            {
                "qdt": "2020-01-02",
                "exp": "2020-01-31",
                "days": 29,
                "cp": "C",
                "k": 100.0,
                "d": 0.5,
                "b": 6.0,
                "a": 6.2,
                "spot": 101.0,
                "ivm": 0.21,
            },
        ]
    )
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
            "spot": "spot_price",
            "ivm": "market_iv",
        }
    )

    structure = StructureSpec(
        name="single_call",
        dte_target=30,
        dte_tolerance=3,
        legs=(LegSpec(option_type=OptionType.CALL, delta_target=0.5),),
    )
    spec = StrategySpec(
        signal=_AlwaysShortSignal(),
        structure_spec=structure,
        lifecycle=LifecycleConfig(rebalance_period=1),
    )
    cfg = BacktestRunConfig(
        account=AccountConfig(initial_capital=10_000.0),
        execution=ExecutionConfig(
            lot_size=1,
            slip_ask=0.0,
            slip_bid=0.0,
            commission_per_leg=0.0,
        ),
    )
    data = OptionsBacktestDataBundle(options=options_raw, options_adapter=adapter)
    bt = Backtester(data=data, strategy=spec, config=cfg)
    trades, mtm = bt.run()

    assert not trades.empty
    assert not mtm.empty

import math

import pandas as pd
import pytest

from volatility_trading.backtesting import HedgeMarketData
from volatility_trading.backtesting.options_engine.contracts.market import (
    HedgeMarketSnapshot,
)


def test_hedge_market_snapshot_from_market_data_resolves_series_values():
    market = HedgeMarketData(
        mid=pd.Series(
            [100.0, 101.0],
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
        bid=pd.Series(
            [99.5, 100.5],
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
        ask=pd.Series(
            [100.5, 101.5],
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        ),
        contract_multiplier=50.0,
    )

    snapshot = HedgeMarketSnapshot.from_market_data(
        hedge_market=market,
        curr_date=pd.Timestamp("2024-01-03"),
    )

    assert snapshot.mid == pytest.approx(101.0)
    assert snapshot.bid == pytest.approx(100.5)
    assert snapshot.ask == pytest.approx(101.5)
    assert snapshot.contract_multiplier == pytest.approx(50.0)


def test_hedge_market_snapshot_from_market_data_handles_missing_values():
    market = HedgeMarketData(
        mid=pd.Series([100.0], index=pd.to_datetime(["2024-01-02"])),
    )

    snapshot = HedgeMarketSnapshot.from_market_data(
        hedge_market=market,
        curr_date=pd.Timestamp("2024-01-10"),
    )

    assert math.isnan(snapshot.mid)
    assert math.isnan(snapshot.bid)
    assert math.isnan(snapshot.ask)
    assert snapshot.contract_multiplier == pytest.approx(1.0)


def test_hedge_market_snapshot_from_market_data_handles_missing_market():
    snapshot = HedgeMarketSnapshot.from_market_data(
        hedge_market=None,
        curr_date=pd.Timestamp("2024-01-10"),
    )

    assert math.isnan(snapshot.mid)
    assert math.isnan(snapshot.bid)
    assert math.isnan(snapshot.ask)
    assert snapshot.contract_multiplier == pytest.approx(1.0)

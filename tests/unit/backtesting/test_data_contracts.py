import pandas as pd
import pytest

from volatility_trading.backtesting import HedgeMarketData


def test_hedge_market_data_accepts_positive_contract_multiplier():
    market = HedgeMarketData(
        mid=pd.Series([100.0], index=pd.to_datetime(["2024-01-02"])),
        contract_multiplier=50.0,
    )
    assert market.contract_multiplier == pytest.approx(50.0)


def test_hedge_market_data_rejects_non_positive_or_non_finite_multiplier():
    mid = pd.Series([100.0], index=pd.to_datetime(["2024-01-02"]))
    with pytest.raises(ValueError, match="contract_multiplier must be finite and > 0"):
        HedgeMarketData(mid=mid, contract_multiplier=0.0)
    with pytest.raises(ValueError, match="contract_multiplier must be finite and > 0"):
        HedgeMarketData(mid=mid, contract_multiplier=float("nan"))

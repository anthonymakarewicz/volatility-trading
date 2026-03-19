import pandas as pd
import pytest

from volatility_trading.backtesting.attribution import to_daily_mtm


def test_to_daily_mtm_uses_trading_dates_without_inserting_weekends():
    raw_mtm = pd.DataFrame(
        {
            "delta_pnl": [5.0, 1.0],
            "net_delta": [2.0, 0.0],
            "delta": [2.0, 0.0],
            "gamma": [0.2, 0.0],
            "vega": [4.0, 0.0],
            "theta": [-1.0, 0.0],
            "S": [100.0, 103.0],
            "iv": [0.20, 0.21],
        },
        index=pd.to_datetime(["2020-01-02", "2020-01-06"]),
    )

    daily = to_daily_mtm(
        raw_mtm,
        initial_capital=100.0,
        trading_dates=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
    )

    expected_index = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    assert daily.index.equals(expected_index)
    assert int((pd.DatetimeIndex(daily.index).dayofweek >= 5).sum()) == 0
    assert daily.loc[pd.Timestamp("2020-01-03"), "delta_pnl"] == pytest.approx(0.0)
    assert daily.loc[pd.Timestamp("2020-01-03"), "equity"] == pytest.approx(105.0)
    assert daily.loc[pd.Timestamp("2020-01-06"), "dt"] == 3
    assert daily.loc[pd.Timestamp("2020-01-06"), "Theta_PnL"] == pytest.approx(-3.0)


def test_to_daily_mtm_uses_factor_columns_for_vol_attribution_when_available():
    raw_mtm = pd.DataFrame(
        {
            "delta_pnl": [0.0, 4.0],
            "net_delta": [0.0, 0.0],
            "delta": [0.0, 0.0],
            "gamma": [0.0, 0.0],
            "vega": [2.0, 2.0],
            "theta": [0.0, 0.0],
            "S": [100.0, 100.0],
            "iv": [0.20, 0.20],
            "factor_iv_level": [20.0, 20.0],
            "factor_exposure_iv_level": [5.0, 5.0],
            "factor_rr_skew": [-11.0, -9.0],
            "factor_exposure_rr_skew": [2.0, 2.0],
        },
        index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
    )

    daily = to_daily_mtm(raw_mtm, initial_capital=100.0)

    assert daily.loc[pd.Timestamp("2020-01-03"), "IV_Level_PnL"] == pytest.approx(0.0)
    assert daily.loc[pd.Timestamp("2020-01-03"), "RR_Skew_PnL"] == pytest.approx(4.0)
    assert daily.loc[pd.Timestamp("2020-01-03"), "Vega_PnL"] == pytest.approx(4.0)
    assert daily.loc[pd.Timestamp("2020-01-03"), "Other_PnL"] == pytest.approx(0.0)

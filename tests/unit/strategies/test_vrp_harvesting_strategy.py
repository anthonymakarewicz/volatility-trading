import pandas as pd
import pytest

from volatility_trading.backtesting import BacktestConfig
from volatility_trading.backtesting.engine import Backtester
from volatility_trading.signals import ShortOnlySignal
from volatility_trading.strategies import VRPHarvestingStrategy


def _run_backtest(options: pd.DataFrame, *, holding_period: int):
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        lot_size=1,
        slip_ask=0.0,
        slip_bid=0.0,
        commission_per_leg=0.0,
    )
    strat = VRPHarvestingStrategy(
        signal=ShortOnlySignal(), holding_period=holding_period
    )
    bt = Backtester(
        data={"options": options, "features": None, "hedge": None},
        strategy=strat,
        config=cfg,
    )
    return bt.run()


def _make_options(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    return df.set_index("trade_date").sort_index()


def test_mtm_delta_pnl_matches_trade_pnl_on_holding_period_exit():
    options = _make_options(
        [
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-02",
                "expiry_date": "2020-01-31",
                "dte": 29,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.8,
                "ask_price": 6.2,
                "spot_price": 101.0,
                "smoothed_iv": 0.22,
            },
            {
                "trade_date": "2020-01-02",
                "expiry_date": "2020-01-31",
                "dte": 29,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.8,
                "ask_price": 6.2,
                "spot_price": 101.0,
                "smoothed_iv": 0.22,
            },
            {
                "trade_date": "2020-01-03",
                "expiry_date": "2020-01-31",
                "dte": 28,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 6.2,
                "ask_price": 6.5,
                "spot_price": 102.0,
                "smoothed_iv": 0.23,
            },
            {
                "trade_date": "2020-01-03",
                "expiry_date": "2020-01-31",
                "dte": 28,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 6.2,
                "ask_price": 6.5,
                "spot_price": 102.0,
                "smoothed_iv": 0.23,
            },
        ]
    )

    trades, mtm = _run_backtest(options, holding_period=2)

    assert len(trades) == 1
    assert trades["pnl"].sum() == pytest.approx(-3.0)
    assert mtm["delta_pnl"].sum() == pytest.approx(trades["pnl"].sum())


def test_unresolved_trade_keeps_mtm_path_instead_of_being_dropped():
    options = _make_options(
        [
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "strike": 100.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            # Matching expiry, but not the held strike, so the position never exits.
            {
                "trade_date": "2020-01-02",
                "expiry_date": "2020-01-31",
                "dte": 29,
                "strike": 105.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 4.8,
                "ask_price": 5.0,
                "spot_price": 105.0,
                "smoothed_iv": 0.21,
            },
            {
                "trade_date": "2020-01-02",
                "expiry_date": "2020-01-31",
                "dte": 29,
                "strike": 105.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 4.8,
                "ask_price": 5.0,
                "spot_price": 105.0,
                "smoothed_iv": 0.21,
            },
            {
                "trade_date": "2020-01-03",
                "expiry_date": "2020-01-31",
                "dte": 28,
                "strike": 105.0,
                "option_type": "P",
                "delta": -0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 4.7,
                "ask_price": 4.9,
                "spot_price": 105.0,
                "smoothed_iv": 0.21,
            },
            {
                "trade_date": "2020-01-03",
                "expiry_date": "2020-01-31",
                "dte": 28,
                "strike": 105.0,
                "option_type": "C",
                "delta": 0.5,
                "gamma": 0.01,
                "vega": 0.10,
                "theta": -0.02,
                "bid_price": 4.7,
                "ask_price": 4.9,
                "spot_price": 105.0,
                "smoothed_iv": 0.21,
            },
        ]
    )

    trades, mtm = _run_backtest(options, holding_period=2)

    assert trades.empty
    assert not mtm.empty
    assert mtm.index.min() == pd.Timestamp("2020-01-01")
    assert mtm.index.max() == pd.Timestamp("2020-01-03")

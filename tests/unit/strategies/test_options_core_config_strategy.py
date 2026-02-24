import pandas as pd
import pytest

from volatility_trading.backtesting import BacktestConfig
from volatility_trading.backtesting.engine import Backtester
from volatility_trading.options import OptionType
from volatility_trading.signals.base_signal import Signal
from volatility_trading.strategies.options_core import (
    LegSpec,
    OptionsStrategyRunner,
    StrategySpec,
    StructureSpec,
)


class DirectionSignal(Signal):
    def __init__(self, *, direction: int):
        super().__init__()
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        self.direction = direction

    def generate_signals(self, data):
        idx = data.index
        if self.direction == 1:
            return pd.DataFrame({"long": True, "short": False}, index=idx)
        return pd.DataFrame({"long": False, "short": True}, index=idx)

    def get_params(self) -> dict:
        return {"direction": self.direction}

    def set_params(self, **kwargs):
        if "direction" in kwargs:
            self.direction = int(kwargs["direction"])


def _make_options() -> pd.DataFrame:
    rows = [
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
            "option_type": "C",
            "delta": 0.5,
            "gamma": 0.01,
            "vega": 0.10,
            "theta": -0.02,
            "bid_price": 6.0,
            "ask_price": 6.2,
            "spot_price": 101.0,
            "smoothed_iv": 0.21,
        },
    ]
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    return df.set_index("trade_date").sort_index()


def _run_strategy(direction: int):
    structure = StructureSpec(
        name="single_call",
        dte_target=30,
        dte_tolerance=3,
        legs=(LegSpec(option_type=OptionType.CALL, delta_target=0.5),),
    )
    spec = StrategySpec(
        name="directional_test",
        signal=DirectionSignal(direction=direction),
        structure_spec=structure,
        rebalance_period=1,
        max_holding_period=None,
    )
    strategy = OptionsStrategyRunner(spec)
    cfg = BacktestConfig(
        initial_capital=10_000.0,
        lot_size=1,
        slip_ask=0.0,
        slip_bid=0.0,
        commission_per_leg=0.0,
    )
    bt = Backtester(
        data={"options": _make_options(), "features": None, "hedge": None},
        strategy=strategy,
        config=cfg,
    )
    return bt.run()


def test_config_strategy_uses_long_direction_for_entry_side():
    trades, _ = _run_strategy(direction=1)

    assert len(trades) == 1
    row = trades.iloc[0]
    leg = row["trade_legs"][0]
    assert leg["side"] == 1
    assert leg["effective_side"] == 1
    assert row["pnl"] == pytest.approx(0.8)


def test_config_strategy_uses_short_direction_for_entry_side():
    trades, _ = _run_strategy(direction=-1)

    assert len(trades) == 1
    row = trades.iloc[0]
    leg = row["trade_legs"][0]
    assert leg["side"] == -1
    assert leg["effective_side"] == -1
    assert row["pnl"] == pytest.approx(-1.2)

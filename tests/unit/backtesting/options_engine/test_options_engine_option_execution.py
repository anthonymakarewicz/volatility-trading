import pandas as pd
import pytest

from volatility_trading.backtesting.options_engine import (
    BidAskFeeOptionExecutionModel,
    MidNoCostOptionExecutionModel,
    OptionExecutionOrder,
    QuoteSnapshot,
)


def _make_quote(*, bid_price: float = 4.0, ask_price: float = 6.0):
    return QuoteSnapshot.from_series(
        pd.Series(
            {
                "option_type": "C",
                "strike": 100.0,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "delta": 0.5,
                "gamma": 0.1,
                "vega": 0.2,
                "theta": -0.3,
            }
        )
    )


def test_option_execution_order_rejects_invalid_trade_side():
    with pytest.raises(ValueError, match="trade_side"):
        OptionExecutionOrder(
            quote=_make_quote(),
            trade_side=0,
            quantity=1.0,
            fee_contracts=1.0,
        )


def test_option_execution_order_rejects_negative_or_nonfinite_inputs():
    with pytest.raises(ValueError, match="quantity"):
        OptionExecutionOrder(
            quote=_make_quote(),
            trade_side=1,
            quantity=-1.0,
            fee_contracts=1.0,
        )
    with pytest.raises(ValueError, match="quantity"):
        OptionExecutionOrder(
            quote=_make_quote(),
            trade_side=1,
            quantity=float("inf"),
            fee_contracts=1.0,
        )
    with pytest.raises(ValueError, match="fee_contracts"):
        OptionExecutionOrder(
            quote=_make_quote(),
            trade_side=1,
            quantity=1.0,
            fee_contracts=-1.0,
        )


def test_mid_no_cost_option_execution_model_fills_at_mid_and_charges_no_cost():
    model = MidNoCostOptionExecutionModel()
    result = model.execute(
        order=OptionExecutionOrder(
            quote=_make_quote(bid_price=4.0, ask_price=6.0),
            trade_side=1,
            quantity=10.0,
            fee_contracts=3.0,
        ),
    )

    assert result.fill_price == pytest.approx(5.0)
    assert result.price_cost == pytest.approx(0.0)
    assert result.fee_cost == pytest.approx(0.0)
    assert result.total_cost == pytest.approx(0.0)


def test_bid_ask_fee_option_execution_model_buy_side_applies_slippage_and_fees():
    model = BidAskFeeOptionExecutionModel(
        slip_ask=0.25,
        slip_bid=0.10,
        commission_per_leg=1.5,
    )
    result = model.execute(
        order=OptionExecutionOrder(
            quote=_make_quote(bid_price=4.0, ask_price=6.0),
            trade_side=1,
            quantity=10.0,
            fee_contracts=2.0,
        ),
    )

    # mid=5.0, buy fill=ask+slip=6.25 => price_cost=(1.25*10)=12.5
    # fee_cost=1.5*2=3.0 => total=15.5
    assert result.fill_price == pytest.approx(6.25)
    assert result.price_cost == pytest.approx(12.5)
    assert result.fee_cost == pytest.approx(3.0)
    assert result.total_cost == pytest.approx(15.5)


def test_bid_ask_fee_option_execution_model_sell_side_applies_bid_and_slippage():
    model = BidAskFeeOptionExecutionModel(
        slip_ask=0.25,
        slip_bid=0.10,
        commission_per_leg=1.5,
    )
    result = model.execute(
        order=OptionExecutionOrder(
            quote=_make_quote(bid_price=4.0, ask_price=6.0),
            trade_side=-1,
            quantity=3.0,
            fee_contracts=0.0,
        ),
    )

    # mid=5.0, sell fill=bid-slip=3.9 => price_cost=(1.1*3)=3.3
    assert result.fill_price == pytest.approx(3.9)
    assert result.price_cost == pytest.approx(3.3)
    assert result.fee_cost == pytest.approx(0.0)
    assert result.total_cost == pytest.approx(3.3)


def test_bid_ask_fee_option_execution_model_charges_fee_even_without_price_quantity():
    model = BidAskFeeOptionExecutionModel(
        slip_ask=0.25,
        slip_bid=0.10,
        commission_per_leg=1.5,
    )
    result = model.execute(
        order=OptionExecutionOrder(
            quote=_make_quote(bid_price=4.0, ask_price=6.0),
            trade_side=1,
            quantity=0.0,
            fee_contracts=2.0,
        ),
    )

    assert result.fill_price == pytest.approx(6.25)
    assert result.price_cost == pytest.approx(0.0)
    assert result.fee_cost == pytest.approx(3.0)
    assert result.total_cost == pytest.approx(3.0)

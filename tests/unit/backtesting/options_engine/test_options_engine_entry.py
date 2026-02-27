import pandas as pd
import pytest

from volatility_trading.backtesting import (
    AccountConfig,
    BacktestRunConfig,
    ExecutionConfig,
)
from volatility_trading.backtesting.options_engine import (
    LegSpec,
    StructureSpec,
    build_entry_intent_from_structure,
    normalize_signals_to_on,
)
from volatility_trading.options import OptionType


def _base_cfg() -> BacktestRunConfig:
    return BacktestRunConfig(
        account=AccountConfig(initial_capital=10_000.0),
        execution=ExecutionConfig(
            lot_size=1,
            slip_ask=0.0,
            slip_bid=0.0,
            commission_per_leg=0.0,
        ),
    )


def _base_chain() -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "option_type": "C",
                "delta": 0.50,
                "strike": 100.0,
                "bid_price": 5.0,
                "ask_price": 5.2,
                "open_interest": 200,
                "volume": 300,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-01-31",
                "dte": 30,
                "option_type": "P",
                "delta": -0.50,
                "strike": 100.0,
                "bid_price": 5.1,
                "ask_price": 5.3,
                "open_interest": 200,
                "volume": 300,
                "spot_price": 100.0,
                "smoothed_iv": 0.20,
            },
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-03-01",
                "dte": 60,
                "option_type": "C",
                "delta": 0.50,
                "strike": 100.0,
                "bid_price": 7.0,
                "ask_price": 7.2,
                "open_interest": 250,
                "volume": 350,
                "spot_price": 100.0,
                "smoothed_iv": 0.22,
            },
            {
                "trade_date": "2020-01-01",
                "expiry_date": "2020-03-01",
                "dte": 60,
                "option_type": "P",
                "delta": -0.50,
                "strike": 100.0,
                "bid_price": 7.1,
                "ask_price": 7.3,
                "open_interest": 250,
                "volume": 350,
                "spot_price": 100.0,
                "smoothed_iv": 0.22,
            },
        ]
    )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    return df.set_index("trade_date")


def test_build_entry_intent_supports_grouped_expiry_selection():
    options = _base_chain()
    structure = StructureSpec(
        name="diag_two_groups",
        dte_target=30,
        dte_tolerance=7,
        legs=(
            LegSpec(
                option_type=OptionType.CALL,
                delta_target=0.50,
                expiry_group="near",
                dte_target=30,
                dte_tolerance=2,
            ),
            LegSpec(
                option_type=OptionType.PUT,
                delta_target=-0.50,
                expiry_group="far",
                dte_target=60,
                dte_tolerance=2,
            ),
        ),
    )

    intent = build_entry_intent_from_structure(
        entry_date=pd.Timestamp("2020-01-01"),
        options=options,
        structure_spec=structure,
        cfg=_base_cfg(),
        side_resolver=lambda _leg: -1,
    )

    assert intent is not None
    assert len(intent.legs) == 2
    expiry0 = intent.legs[0].quote.expiry_date
    expiry1 = intent.legs[1].quote.expiry_date
    assert expiry0 is not None
    assert expiry1 is not None
    assert pd.Timestamp(expiry0) == pd.Timestamp("2020-01-31")
    assert pd.Timestamp(expiry1) == pd.Timestamp("2020-03-01")


def test_build_entry_intent_all_or_none_rejects_partial_fill():
    options = _base_chain()
    options = options[options["expiry_date"] != pd.Timestamp("2020-03-01")]
    structure = StructureSpec(
        name="two_groups_all_or_none",
        dte_target=30,
        dte_tolerance=7,
        legs=(
            LegSpec(
                option_type=OptionType.CALL,
                delta_target=0.50,
                expiry_group="near",
                dte_target=30,
                dte_tolerance=2,
            ),
            LegSpec(
                option_type=OptionType.PUT,
                delta_target=-0.50,
                expiry_group="far",
                dte_target=60,
                dte_tolerance=2,
            ),
        ),
        fill_policy="all_or_none",
    )

    intent = build_entry_intent_from_structure(
        entry_date=pd.Timestamp("2020-01-01"),
        options=options,
        structure_spec=structure,
        cfg=_base_cfg(),
        side_resolver=lambda _leg: -1,
    )
    assert intent is None


def test_build_entry_intent_min_ratio_allows_partial_fill():
    options = _base_chain()
    options = options[options["expiry_date"] != pd.Timestamp("2020-03-01")]
    structure = StructureSpec(
        name="two_groups_partial_ok",
        dte_target=30,
        dte_tolerance=7,
        legs=(
            LegSpec(
                option_type=OptionType.CALL,
                delta_target=0.50,
                expiry_group="near",
                dte_target=30,
                dte_tolerance=2,
            ),
            LegSpec(
                option_type=OptionType.PUT,
                delta_target=-0.50,
                expiry_group="far",
                dte_target=60,
                dte_tolerance=2,
            ),
        ),
        fill_policy="min_ratio",
        min_fill_ratio=0.5,
    )

    intent = build_entry_intent_from_structure(
        entry_date=pd.Timestamp("2020-01-01"),
        options=options,
        structure_spec=structure,
        cfg=_base_cfg(),
        side_resolver=lambda _leg: -1,
    )
    assert intent is not None
    assert len(intent.legs) == 1
    expiry0 = intent.legs[0].quote.expiry_date
    assert expiry0 is not None
    assert pd.Timestamp(expiry0) == pd.Timestamp("2020-01-31")


def test_normalize_signals_to_on_rejects_on_only_inputs():
    signals = pd.Series(
        [True, False, True],
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        name="on",
    )
    with pytest.raises(ValueError, match="entry_direction"):
        normalize_signals_to_on(signals)


def test_normalize_signals_to_on_builds_direction_from_long_short_columns():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    signals = pd.DataFrame(
        {
            "long": [True, False, False],
            "short": [False, True, False],
        },
        index=idx,
    )
    out = normalize_signals_to_on(signals)
    assert list(out["on"]) == [True, True, False]
    assert list(out["entry_direction"]) == [1, -1, 0]

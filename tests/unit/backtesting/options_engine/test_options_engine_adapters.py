import pandas as pd
import pytest

from volatility_trading.backtesting.options_engine import (
    LegSpec,
    QuoteSnapshot,
    StructureSpec,
    normalize_chain_option_type,
    option_type_to_chain_label,
    quote_to_option_leg,
    quote_to_option_spec,
    time_to_expiry_years,
)
from volatility_trading.options import OptionType, PositionSide


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("C", OptionType.CALL),
        ("P", OptionType.PUT),
        ("call", OptionType.CALL),
        ("put", OptionType.PUT),
        (OptionType.CALL, OptionType.CALL),
        (OptionType.PUT, OptionType.PUT),
    ],
)
def test_normalize_chain_option_type(raw, expected):
    assert normalize_chain_option_type(raw) == expected


def test_option_type_to_chain_label():
    assert option_type_to_chain_label(OptionType.CALL) == "C"
    assert option_type_to_chain_label(OptionType.PUT) == "P"


def test_time_to_expiry_years_prefers_yte_then_dte_then_calendar():
    entry = pd.Timestamp("2020-01-01")
    expiry = pd.Timestamp("2020-01-31")

    yte = time_to_expiry_years(
        entry_date=entry,
        expiry_date=expiry,
        quote_yte=0.2,
        quote_dte=30,
    )
    dte = time_to_expiry_years(
        entry_date=entry,
        expiry_date=expiry,
        quote_yte=float("nan"),
        quote_dte=30,
    )
    calendar = time_to_expiry_years(
        entry_date=entry,
        expiry_date=expiry,
        quote_yte=None,
        quote_dte=0,
    )

    assert yte == pytest.approx(0.2)
    assert dte == pytest.approx(30 / 365.0)
    assert calendar == pytest.approx(30 / 365.0)


def test_quote_to_option_spec_and_leg():
    quote = QuoteSnapshot.from_series(
        pd.Series(
            {
                "strike": 100.0,
                "yte": 0.15,
                "dte": 54,
                "option_type": "P",
            }
        )
    )
    entry = pd.Timestamp("2020-01-01")
    expiry = pd.Timestamp("2020-02-01")

    spec = quote_to_option_spec(
        quote=quote,
        entry_date=entry,
        expiry_date=expiry,
    )
    assert spec.strike == pytest.approx(100.0)
    assert spec.time_to_expiry == pytest.approx(0.15)
    assert spec.option_type == OptionType.PUT

    leg = quote_to_option_leg(
        quote=quote,
        entry_date=entry,
        expiry_date=expiry,
        entry_price=4.2,
        side=-1,
        contract_multiplier=100,
    )
    assert leg.entry_price == pytest.approx(4.2)
    assert leg.side == PositionSide.SHORT
    assert leg.contract_multiplier == pytest.approx(100)
    assert leg.spec.option_type == OptionType.PUT


def test_structure_spec_requires_non_empty_legs():
    with pytest.raises(ValueError, match="legs must not be empty"):
        StructureSpec(name="empty", legs=())


def test_leg_spec_rejects_zero_weight():
    with pytest.raises(ValueError, match="weight must be non-zero"):
        LegSpec(
            option_type=OptionType.CALL,
            delta_target=0.5,
            weight=0,
        )

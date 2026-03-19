import pytest

from volatility_trading.backtesting.options_engine import RiskReversalFactorModel
from volatility_trading.backtesting.options_engine.contracts.market import QuoteSnapshot
from volatility_trading.backtesting.options_engine.contracts.structures import (
    LegSelection,
    LegSpec,
    StructureSpec,
)
from volatility_trading.options import OptionType, PositionSide


def _quote(*, option_type: str, market_iv: float, vega: float) -> QuoteSnapshot:
    return QuoteSnapshot(
        option_type_label=option_type,
        strike=100.0,
        bid_price=1.0,
        ask_price=1.1,
        delta=0.0,
        gamma=0.0,
        vega=vega,
        theta=0.0,
        market_iv=market_iv,
    )


def test_risk_reversal_factor_model_rejects_non_rr_structure() -> None:
    model = RiskReversalFactorModel()
    structure = StructureSpec(
        name="call_spread",
        legs=(
            LegSpec(option_type=OptionType.CALL, delta_target=0.25),
            LegSpec(option_type=OptionType.CALL, delta_target=0.50),
        ),
    )

    with pytest.raises(ValueError, match="exactly one call leg and one put leg"):
        model.validate_structure(structure_spec=structure)


def test_risk_reversal_factor_model_snapshot_returns_iv_level_and_rr_skew() -> None:
    model = RiskReversalFactorModel()
    put_leg = LegSelection(
        spec=LegSpec(option_type=OptionType.PUT, delta_target=-0.25),
        quote=_quote(option_type="P", market_iv=0.30, vega=0.12),
        side=PositionSide.SHORT,
        entry_price=1.0,
    )
    call_leg = LegSelection(
        spec=LegSpec(option_type=OptionType.CALL, delta_target=0.25),
        quote=_quote(option_type="C", market_iv=0.19, vega=0.10),
        side=PositionSide.LONG,
        entry_price=1.0,
    )

    snapshot = model.snapshot(
        legs=(put_leg, call_leg),
        leg_quotes=(put_leg.quote, call_leg.quote),
        option_contract_multiplier=100.0,
        contracts=2,
    )

    assert snapshot.values == {
        "iv_level": pytest.approx(24.5),
        "rr_skew": pytest.approx(-11.0),
    }
    assert snapshot.exposures == {
        "iv_level": pytest.approx(-4.0),
        "rr_skew": pytest.approx(22.0),
    }

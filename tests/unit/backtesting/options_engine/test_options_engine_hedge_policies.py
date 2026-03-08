import pytest

from volatility_trading.backtesting import ExecutionConfig, HedgeExecutionConfig
from volatility_trading.backtesting.options_engine.lifecycle.hedge_policies import (
    HedgeBandContext,
    evaluate_band_target,
)
from volatility_trading.backtesting.options_engine.specs import (
    DeltaHedgePolicy,
    FixedDeltaBandModel,
    HedgeTriggerPolicy,
    WWDeltaBandModel,
)


def _make_band_context(
    *,
    option_gamma: float = 0.1,
    option_volatility: float = 0.2,
    hedge_price: float = 100.0,
    fee_bps: float = 10.0,
) -> HedgeBandContext:
    return HedgeBandContext(
        option_gamma=option_gamma,
        option_volatility=option_volatility,
        hedge_price=hedge_price,
        execution=ExecutionConfig(
            hedge=HedgeExecutionConfig(
                fee_bps=fee_bps,
            )
        ),
    )


def test_fixed_center_mode_keeps_center_target_inside_band():
    policy = DeltaHedgePolicy(
        enabled=True,
        target_net_delta=0.0,
        trigger=HedgeTriggerPolicy(
            band_model=FixedDeltaBandModel(half_width_abs=0.1),
        ),
        rebalance_to="center",
    )

    decision = evaluate_band_target(
        policy=policy,
        center_target_net_delta=0.0,
        net_delta_before=0.05,
        context=_make_band_context(),
    )

    assert decision.target_net_delta == pytest.approx(0.0)
    assert decision.delta_trigger is False
    assert decision.band_half_width == pytest.approx(0.1)
    assert decision.band_lower == pytest.approx(-0.1)
    assert decision.band_upper == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("net_delta_before", "expected_target", "expected_trigger"),
    [
        (-0.25, -0.1, True),
        (0.04, 0.04, False),
        (0.25, 0.1, True),
    ],
)
def test_fixed_nearest_boundary_modes_match_boundary_policy(
    net_delta_before: float,
    expected_target: float,
    expected_trigger: bool,
):
    policy = DeltaHedgePolicy(
        enabled=True,
        target_net_delta=0.0,
        trigger=HedgeTriggerPolicy(
            band_model=FixedDeltaBandModel(half_width_abs=0.1),
        ),
        rebalance_to="nearest_boundary",
    )

    decision = evaluate_band_target(
        policy=policy,
        center_target_net_delta=0.0,
        net_delta_before=net_delta_before,
        context=_make_band_context(),
    )

    assert decision.target_net_delta == pytest.approx(expected_target)
    assert decision.delta_trigger is expected_trigger
    assert decision.band_half_width == pytest.approx(0.1)
    assert decision.band_lower == pytest.approx(-0.1)
    assert decision.band_upper == pytest.approx(0.1)


def test_no_band_model_returns_center_target_and_none_trigger():
    policy = DeltaHedgePolicy(
        enabled=True,
        target_net_delta=0.0,
        trigger=HedgeTriggerPolicy(
            band_model=None,
            rebalance_every_n_days=5,
        ),
        rebalance_to="center",
    )

    decision = evaluate_band_target(
        policy=policy,
        center_target_net_delta=0.25,
        net_delta_before=2.0,
        context=_make_band_context(),
    )

    assert decision.target_net_delta == pytest.approx(0.25)
    assert decision.delta_trigger is None
    assert decision.band_half_width is None
    assert decision.band_lower is None
    assert decision.band_upper is None


def test_ww_band_clamps_to_min_and_max():
    ww_min = WWDeltaBandModel(
        calibration_c=1.0,
        min_band_abs=0.2,
        max_band_abs=1.0,
    )
    policy_min = DeltaHedgePolicy(
        enabled=True,
        target_net_delta=0.0,
        trigger=HedgeTriggerPolicy(band_model=ww_min),
        rebalance_to="nearest_boundary",
    )
    decision_min = evaluate_band_target(
        policy=policy_min,
        center_target_net_delta=0.0,
        net_delta_before=0.0,
        context=_make_band_context(
            option_gamma=20.0,
            option_volatility=1.0,
            hedge_price=10_000.0,
            fee_bps=1.0,
        ),
    )
    assert decision_min.band_half_width == pytest.approx(0.2)

    ww_max = WWDeltaBandModel(
        calibration_c=1.0,
        min_band_abs=0.0,
        max_band_abs=1.5,
    )
    policy_max = DeltaHedgePolicy(
        enabled=True,
        target_net_delta=0.0,
        trigger=HedgeTriggerPolicy(band_model=ww_max),
        rebalance_to="nearest_boundary",
    )
    decision_max = evaluate_band_target(
        policy=policy_max,
        center_target_net_delta=0.0,
        net_delta_before=0.0,
        context=_make_band_context(
            option_gamma=1e-8,
            option_volatility=1e-6,
            hedge_price=1e-8,
            fee_bps=100.0,
        ),
    )
    assert decision_max.band_half_width == pytest.approx(1.5)


def test_ww_fee_override_takes_precedence_over_execution_fee_bps():
    policy_exec_fee = DeltaHedgePolicy(
        enabled=True,
        target_net_delta=0.0,
        trigger=HedgeTriggerPolicy(
            band_model=WWDeltaBandModel(
                calibration_c=1.0,
                min_band_abs=0.0,
                max_band_abs=10.0,
                fee_bps_override=None,
            ),
        ),
        rebalance_to="nearest_boundary",
    )
    policy_override_fee = DeltaHedgePolicy(
        enabled=True,
        target_net_delta=0.0,
        trigger=HedgeTriggerPolicy(
            band_model=WWDeltaBandModel(
                calibration_c=1.0,
                min_band_abs=0.0,
                max_band_abs=10.0,
                fee_bps_override=5.0,
            ),
        ),
        rebalance_to="nearest_boundary",
    )
    context = _make_band_context(
        option_gamma=0.1,
        option_volatility=0.2,
        hedge_price=100.0,
        fee_bps=50.0,
    )

    decision_exec_fee = evaluate_band_target(
        policy=policy_exec_fee,
        center_target_net_delta=0.0,
        net_delta_before=0.0,
        context=context,
    )
    decision_override_fee = evaluate_band_target(
        policy=policy_override_fee,
        center_target_net_delta=0.0,
        net_delta_before=0.0,
        context=context,
    )

    assert decision_exec_fee.band_half_width is not None
    assert decision_override_fee.band_half_width is not None
    assert decision_override_fee.band_half_width < decision_exec_fee.band_half_width

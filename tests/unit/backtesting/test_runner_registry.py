import pytest

from volatility_trading.backtesting.runner.registry import (
    available_signal_names,
    available_strategy_preset_names,
    build_signal,
    build_strategy_preset,
)
from volatility_trading.backtesting.runner.types import (
    NamedSignalSpec,
    NamedStrategyPresetSpec,
)
from volatility_trading.signals import LongOnlySignal, ShortOnlySignal, ZScoreSignal


def test_available_runner_registry_names_are_stable() -> None:
    assert available_signal_names() == ("long_only", "short_only", "zscore")
    assert available_strategy_preset_names() == (
        "skew_mispricing",
        "vrp_harvesting",
    )


def test_build_signal_builds_short_only_signal() -> None:
    signal = build_signal(NamedSignalSpec(name="short_only"))

    assert isinstance(signal, ShortOnlySignal)


def test_build_signal_builds_zscore_signal_with_params() -> None:
    signal = build_signal(
        NamedSignalSpec(
            name="zscore",
            params={"window": 21, "entry": 1.75, "exit": 0.4},
        )
    )

    assert isinstance(signal, ZScoreSignal)
    assert signal.window == 21
    assert signal.entry == pytest.approx(1.75)
    assert signal.exit == pytest.approx(0.4)


def test_build_signal_rejects_unknown_name() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown signal name 'missing'",
    ):
        build_signal(NamedSignalSpec(name="missing"))


def test_build_signal_rejects_invalid_params() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid parameters for signal 'short_only'",
    ):
        build_signal(
            NamedSignalSpec(name="short_only", params={"window": 10}),
        )


def test_named_signal_spec_rejects_blank_name() -> None:
    with pytest.raises(ValueError, match="signal name must be a non-empty string"):
        NamedSignalSpec(name="   ")


def test_build_strategy_preset_builds_vrp_strategy_with_nested_signal() -> None:
    strategy = build_strategy_preset(
        NamedStrategyPresetSpec(
            name="vrp_harvesting",
            signal=NamedSignalSpec(name="long_only"),
            params={
                "rebalance_period": 10,
                "max_holding_period": 30,
                "risk_budget_pct": 0.02,
                "margin_budget_pct": 0.25,
            },
        )
    )

    assert strategy.name == "vrp_harvesting"
    assert isinstance(strategy.signal, LongOnlySignal)
    assert strategy.lifecycle.rebalance_period == 10
    assert strategy.lifecycle.max_holding_period == 30
    assert strategy.sizing.margin_budget_pct == pytest.approx(0.25)
    assert strategy.sizing.risk_sizer is not None
    assert strategy.sizing.risk_sizer.risk_budget_pct == pytest.approx(0.02)


def test_build_strategy_preset_requires_signal_when_preset_has_no_default() -> None:
    with pytest.raises(
        ValueError,
        match="Strategy preset 'vrp_harvesting' requires a signal configuration",
    ):
        build_strategy_preset(
            NamedStrategyPresetSpec(
                name="vrp_harvesting",
            )
        )


def test_build_strategy_preset_builds_skew_strategy_with_default_signal() -> None:
    strategy = build_strategy_preset(
        NamedStrategyPresetSpec(
            name="skew_mispricing",
            params={
                "target_dte": 60,
                "delta_target_abs": 0.30,
            },
        )
    )

    assert strategy.name == "skew_mispricing"
    assert strategy.structure_spec.name == "risk_reversal"
    assert strategy.structure_spec.dte_target == 60
    assert strategy.structure_spec.legs[0].delta_target == pytest.approx(-0.30)
    assert strategy.structure_spec.legs[1].delta_target == pytest.approx(0.30)


def test_build_strategy_preset_allows_skew_signal_override() -> None:
    strategy = build_strategy_preset(
        NamedStrategyPresetSpec(
            name="skew_mispricing",
            signal=NamedSignalSpec(name="long_only"),
        )
    )

    assert isinstance(strategy.signal, LongOnlySignal)


def test_build_strategy_preset_rejects_unknown_name() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown strategy preset name 'missing'",
    ):
        build_strategy_preset(
            NamedStrategyPresetSpec(
                name="missing",
                signal=NamedSignalSpec(name="short_only"),
            )
        )


def test_build_strategy_preset_rejects_invalid_params() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid parameters for strategy preset 'vrp_harvesting'",
    ):
        build_strategy_preset(
            NamedStrategyPresetSpec(
                name="vrp_harvesting",
                signal=NamedSignalSpec(name="short_only"),
                params={"signal": "do_not_allow"},
            )
        )


def test_build_skew_strategy_preset_rejects_invalid_params() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid parameters for strategy preset 'skew_mispricing'",
    ):
        build_strategy_preset(
            NamedStrategyPresetSpec(
                name="skew_mispricing",
                params={"unknown_param": 123},
            )
        )


def test_named_strategy_preset_spec_rejects_blank_name() -> None:
    with pytest.raises(
        ValueError,
        match="strategy preset name must be a non-empty string",
    ):
        NamedStrategyPresetSpec(
            name=" ",
            signal=NamedSignalSpec(name="short_only"),
        )

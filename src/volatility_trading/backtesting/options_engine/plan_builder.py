"""Compile ``StrategySpec`` into executable single-position plans."""

from __future__ import annotations

import pandas as pd

from ..types import BacktestConfig, DataMapping
from .contracts import SinglePositionExecutionPlan, SinglePositionHooks
from .entry import build_entry_intent_from_structure, normalize_signals_to_on
from .lifecycle import PositionLifecycleEngine
from .outputs import build_options_backtest_outputs
from .sizing import SizingRequest, size_entry_intent
from .specs import StrategySpec
from .state import PositionEntrySetup


def build_options_execution_plan(
    *,
    spec: StrategySpec,
    data: DataMapping,
    config: BacktestConfig,
    capital: float,
) -> SinglePositionExecutionPlan:
    """Compile one options ``StrategySpec`` into an engine execution plan."""
    options = data[spec.options_data_key]
    features = data.get(spec.features_data_key)
    hedge = data.get(spec.hedge_data_key)

    signal_input = spec.signal_input_builder(options, features, hedge)
    signals = spec.signal.generate_signals(signal_input)
    if spec.filters:
        filter_ctx = spec.filter_context_builder(options, features, hedge)
        for filt in spec.filters:
            signals = filt.apply(signals, filter_ctx)

    sig_df = normalize_signals_to_on(signals, strategy_name=spec.name)
    options = options.sort_index()
    trading_dates = sorted(options.index.unique())
    direction_by_date = sig_df["entry_direction"].groupby(level=0).last().astype(int)
    active_signal_dates = set(direction_by_date[direction_by_date != 0].index)
    lifecycle_engine = _build_lifecycle_engine(spec)

    hooks = SinglePositionHooks(
        mark_open_position=lambda position, curr_date, equity_running: (
            lifecycle_engine.mark_position(
                position=position,
                curr_date=curr_date,
                options=options,
                cfg=config,
                equity_running=equity_running,
            )
        ),
        prepare_entry=lambda entry_date, equity_running: _prepare_entry_setup(
            spec=spec,
            entry_date=entry_date,
            entry_direction=int(direction_by_date.get(entry_date, 0)),
            options=options,
            features=features,
            equity_running=equity_running,
            cfg=config,
        ),
        open_position=lambda setup, equity_running: lifecycle_engine.open_position(
            setup=setup,
            cfg=config,
            equity_running=equity_running,
        ),
        can_reenter_same_day=lambda trade_rows: (
            spec.reentry_policy.allow_from_trade_records(trade_rows)
        ),
    )
    return SinglePositionExecutionPlan(
        trading_dates=trading_dates,
        active_signal_dates=active_signal_dates,
        initial_equity=float(capital),
        hooks=hooks,
        build_outputs=build_options_backtest_outputs,
    )


def _prepare_entry_setup(
    *,
    spec: StrategySpec,
    entry_date: pd.Timestamp,
    entry_direction: int,
    options: pd.DataFrame,
    features: pd.DataFrame | None,
    equity_running: float,
    cfg: BacktestConfig,
) -> PositionEntrySetup | None:
    """Build one structure entry setup for one date."""
    if entry_direction == 0:
        return None

    intent = build_entry_intent_from_structure(
        entry_date=entry_date,
        options=options,
        structure_spec=spec.structure_spec,
        cfg=cfg,
        side_resolver=lambda leg_spec: spec.side_resolver(
            leg_spec,
            entry_direction,
        ),
        features=features,
        fallback_iv_feature_col=spec.fallback_iv_feature_col,
    )
    if intent is None:
        return None

    if intent.entry_state is None:
        spot_entry = float("nan")
        iv_entry = float("nan")
    else:
        spot_entry = float(intent.entry_state.spot)
        iv_entry = float(intent.entry_state.volatility)
    sizing = size_entry_intent(
        SizingRequest(
            intent=intent,
            lot_size=cfg.lot_size,
            spot=spot_entry,
            volatility=iv_entry,
            equity=float(equity_running),
            pricer=spec.pricer,
            scenario_generator=spec.scenario_generator,
            risk_estimator=spec.risk_estimator,
            risk_sizer=spec.risk_sizer,
            margin_model=spec.margin_model,
            margin_budget_pct=spec.margin_budget_pct,
            min_contracts=spec.min_contracts,
            max_contracts=spec.max_contracts,
        )
    )
    if sizing.contracts <= 0:
        return None

    return PositionEntrySetup(
        intent=intent,
        contracts=int(sizing.contracts),
        risk_per_contract=sizing.risk_per_contract,
        risk_worst_scenario=sizing.risk_scenario,
        margin_per_contract=sizing.margin_per_contract,
    )


def _build_lifecycle_engine(spec: StrategySpec) -> PositionLifecycleEngine:
    """Build lifecycle engine configured from one strategy spec."""
    return PositionLifecycleEngine(
        rebalance_period=spec.rebalance_period,
        max_holding_period=spec.max_holding_period,
        exit_rule_set=spec.exit_rule_set,
        margin_policy=spec.margin_policy,
        margin_model=spec.margin_model,
        pricer=spec.pricer,
    )

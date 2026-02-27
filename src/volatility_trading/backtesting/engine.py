from __future__ import annotations

import pandas as pd

from .config import BacktestRunConfig
from .data_contracts import OptionsBacktestDataBundle
from .options_engine.contracts import SinglePositionExecutionPlan, SinglePositionHooks
from .options_engine.contracts.records import MtmRecord, TradeRecord
from .options_engine.contracts.runtime import OpenPosition
from .options_engine.plan_builder import build_options_execution_plan
from .options_engine.specs import StrategySpec


def _record_delta_pnl(record: MtmRecord) -> float:
    """Extract numeric ``delta_pnl`` from one typed MTM record."""
    return float(record.delta_pnl)


def run_backtest_execution_plan(
    plan: SinglePositionExecutionPlan,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one compiled single-position plan and return strategy outputs."""
    trades: list[TradeRecord] = []
    mtm_records: list[MtmRecord] = []
    equity_running = float(plan.initial_equity)
    open_position: OpenPosition | None = None
    hooks: SinglePositionHooks = plan.hooks

    for curr_date in plan.trading_dates:
        if open_position is not None:
            step_result = hooks.mark_open_position(
                open_position,
                curr_date,
                equity_running,
            )
            open_position = step_result.position
            mtm_record = step_result.mtm_record
            trade_rows = step_result.trade_rows
            mtm_records.append(mtm_record)
            trades.extend(trade_rows)
            equity_running += _record_delta_pnl(mtm_record)

            if open_position is not None:
                continue
            if not hooks.can_reenter_same_day(trade_rows):
                continue

        if curr_date not in plan.active_signal_dates:
            continue

        setup = hooks.prepare_entry(curr_date, equity_running)
        if setup is None:
            continue

        open_position, entry_record = hooks.open_position(setup, equity_running)
        mtm_records.append(entry_record)
        equity_running += _record_delta_pnl(entry_record)

    return plan.build_outputs(trades, mtm_records, plan.initial_equity)


class Backtester:
    def __init__(
        self,
        data: OptionsBacktestDataBundle,
        strategy: StrategySpec,
        config: BacktestRunConfig,
    ):
        """Initialize one options backtester run with typed data/config contracts."""
        self.data = data
        self.strategy = strategy
        self.config = config

    def run(self):
        current_capital = float(self.config.account.initial_capital)
        plan = build_options_execution_plan(
            spec=self.strategy,
            data=self.data,
            config=self.config,
            capital=current_capital,
        )
        trades, mtm = run_backtest_execution_plan(plan)

        return trades, mtm

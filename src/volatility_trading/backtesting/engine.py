from __future__ import annotations

import logging

import pandas as pd

from .config import BacktestRunConfig
from .data_contracts import OptionsBacktestDataBundle
from .options_engine.contracts import SinglePositionExecutionPlan, SinglePositionHooks
from .options_engine.contracts.records import MtmRecord, TradeRecord
from .options_engine.contracts.runtime import OpenPosition
from .options_engine.plan_builder import build_options_execution_plan
from .options_engine.specs import StrategySpec

logger = logging.getLogger(__name__)


def _record_delta_pnl(record: MtmRecord) -> float:
    """Extract numeric ``delta_pnl`` from one typed MTM record."""
    return float(record.delta_pnl)


def run_backtest_execution_plan(
    plan: SinglePositionExecutionPlan,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one compiled single-position plan and return strategy outputs."""
    logger.info(
        "Starting execution plan run: %d trading dates, %d active signal dates",
        len(plan.trading_dates),
        len(plan.active_signal_dates),
    )
    trades: list[TradeRecord] = []
    mtm_records: list[MtmRecord] = []
    equity_running = float(plan.initial_equity)
    open_position: OpenPosition | None = None
    hooks: SinglePositionHooks = plan.hooks

    for curr_date in plan.trading_dates:
        logger.debug(
            "Processing date=%s open_position=%s equity=%.6f",
            curr_date,
            open_position is not None,
            equity_running,
        )
        if open_position is not None:
            try:
                step_result = hooks.mark_open_position(
                    open_position,
                    curr_date,
                    equity_running,
                )
            except Exception:
                logger.exception(
                    "Failed to mark open position on %s at equity=%.6f",
                    curr_date,
                    equity_running,
                )
                raise
            open_position = step_result.position
            mtm_record = step_result.mtm_record
            trade_rows = step_result.trade_rows
            mtm_records.append(mtm_record)
            trades.extend(trade_rows)
            equity_running += _record_delta_pnl(mtm_record)
            logger.debug(
                "Marked position on %s: delta_pnl=%.6f trade_rows=%d open_after_mark=%s",
                curr_date,
                _record_delta_pnl(mtm_record),
                len(trade_rows),
                open_position is not None,
            )

            if open_position is None and trade_rows:
                logger.info(
                    "Position closed on %s (%d trade row(s))",
                    curr_date,
                    len(trade_rows),
                )

            if open_position is not None:
                continue
            if not hooks.can_reenter_same_day(trade_rows):
                logger.debug(
                    "Same-day reentry blocked by policy on %s after close",
                    curr_date,
                )
                continue

        if curr_date not in plan.active_signal_dates:
            logger.debug("No active signal on %s", curr_date)
            continue

        try:
            setup = hooks.prepare_entry(curr_date, equity_running)
        except Exception:
            logger.exception(
                "Failed to prepare entry on %s at equity=%.6f",
                curr_date,
                equity_running,
            )
            raise
        if setup is None:
            logger.debug("No valid entry setup on %s", curr_date)
            continue

        try:
            open_position, entry_record = hooks.open_position(setup, equity_running)
        except Exception:
            logger.exception(
                "Failed to open position on %s at equity=%.6f",
                curr_date,
                equity_running,
            )
            raise
        mtm_records.append(entry_record)
        equity_running += _record_delta_pnl(entry_record)
        logger.info(
            "Opened position on %s: entry_delta_pnl=%.6f equity=%.6f",
            curr_date,
            _record_delta_pnl(entry_record),
            equity_running,
        )

    logger.info(
        "Building outputs: trades=%d mtm_records=%d final_equity=%.6f",
        len(trades),
        len(mtm_records),
        equity_running,
    )
    outputs = plan.build_outputs(trades, mtm_records, plan.initial_equity)
    logger.info("Execution plan run complete")
    return outputs


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
        logger.info(
            "Starting backtest run strategy=%s initial_capital=%.2f",
            self.strategy.name,
            float(self.config.account.initial_capital),
        )
        current_capital = float(self.config.account.initial_capital)
        try:
            plan = build_options_execution_plan(
                spec=self.strategy,
                data=self.data,
                config=self.config,
                capital=current_capital,
            )
        except Exception:
            logger.exception("Failed to build execution plan")
            raise
        logger.info(
            "Execution plan built: trading_dates=%d active_signal_dates=%d",
            len(plan.trading_dates),
            len(plan.active_signal_dates),
        )

        try:
            trades, mtm = run_backtest_execution_plan(plan)
        except Exception:
            logger.exception("Backtest execution failed")
            raise

        if trades.empty:
            logger.warning("Backtest completed with empty trades output")
        if mtm.empty:
            logger.warning("Backtest completed with empty mtm output")
        logger.info(
            "Backtest completed: trades_rows=%d mtm_rows=%d",
            len(trades),
            len(mtm),
        )
        return trades, mtm

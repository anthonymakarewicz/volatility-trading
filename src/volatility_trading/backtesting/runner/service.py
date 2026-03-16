"""Pure orchestration service for config-driven backtest workflow runs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from volatility_trading.backtesting.attribution import to_daily_mtm
from volatility_trading.backtesting.engine import Backtester
from volatility_trading.backtesting.reporting import (
    BacktestReportBundle,
    build_backtest_report_bundle,
    save_backtest_report_bundle,
)

from .assembly import ResolvedWorkflowInputs, assemble_workflow_inputs
from .config_parser import parse_workflow_config
from .serialization import build_report_config_payload
from .workflow_types import BacktestWorkflowSpec

logger = logging.getLogger(__name__)


def _resolve_workflow_trading_dates(
    resolved: ResolvedWorkflowInputs,
    *,
    raw_mtm_index: pd.Index | None = None,
) -> pd.DatetimeIndex:
    """Return the run-filtered trading-session dates implied by resolved inputs."""
    options = resolved.data.options_frame
    if options.empty:
        trading_dates = pd.DatetimeIndex([])
    else:
        trading_dates = pd.DatetimeIndex(options.index.unique())
    if resolved.run_config.start_date is not None:
        trading_dates = trading_dates[
            trading_dates >= pd.Timestamp(resolved.run_config.start_date)
        ]
    if resolved.run_config.end_date is not None:
        trading_dates = trading_dates[
            trading_dates <= pd.Timestamp(resolved.run_config.end_date)
        ]
    if raw_mtm_index is not None:
        raw_dates = pd.DatetimeIndex(pd.to_datetime(raw_mtm_index))
        trading_dates = trading_dates.union(raw_dates)
    return trading_dates.sort_values().drop_duplicates().astype("datetime64[ns]")


@dataclass(frozen=True)
class BacktestWorkflowRunResult:
    """Concrete outputs returned by one workflow-service execution."""

    workflow: BacktestWorkflowSpec
    resolved: ResolvedWorkflowInputs
    trades: pd.DataFrame
    mtm: pd.DataFrame
    daily_mtm: pd.DataFrame
    report_bundle: BacktestReportBundle | None
    report_dir: Path | None


def run_backtest_workflow(
    workflow: BacktestWorkflowSpec,
) -> BacktestWorkflowRunResult:
    """Run one typed workflow spec through assembly, engine, and reporting."""
    logger.info("Running backtest workflow strategy=%s", workflow.strategy.name)
    resolved = assemble_workflow_inputs(workflow)
    backtester = Backtester(
        data=resolved.data,
        strategy=resolved.strategy,
        config=resolved.run_config,
    )
    trades, mtm = backtester.run()

    if mtm.empty:
        logger.warning("Workflow completed with empty MTM output; skipping reporting")
        return BacktestWorkflowRunResult(
            workflow=workflow,
            resolved=resolved,
            trades=trades,
            mtm=mtm,
            daily_mtm=pd.DataFrame(),
            report_bundle=None,
            report_dir=None,
        )

    daily_mtm = to_daily_mtm(
        mtm,
        resolved.run_config.account.initial_capital,
        trading_dates=_resolve_workflow_trading_dates(
            resolved,
            raw_mtm_index=mtm.index,
        ),
    )
    report_bundle = build_backtest_report_bundle(
        trades=trades,
        mtm_daily=daily_mtm,
        run_config=build_report_config_payload(workflow, resolved),
        strategy_name=resolved.strategy.name,
        benchmark=resolved.benchmark,
        benchmark_name=resolved.benchmark_name,
        run_id=workflow.reporting.run_id,
        include_dashboard_plot=workflow.reporting.include_dashboard_plot,
        include_component_plots=workflow.reporting.include_component_plots,
        risk_free_rate=resolved.risk_free_rate,
    )
    report_dir = None
    if workflow.reporting.save_report_bundle:
        report_dir = save_backtest_report_bundle(
            report_bundle,
            output_root=workflow.reporting.output_root,
        )
    return BacktestWorkflowRunResult(
        workflow=workflow,
        resolved=resolved,
        trades=trades,
        mtm=mtm,
        daily_mtm=daily_mtm,
        report_bundle=report_bundle,
        report_dir=report_dir,
    )


def run_backtest_workflow_config(
    config: Mapping[str, Any],
) -> BacktestWorkflowRunResult:
    """Parse one config mapping and execute the resulting typed workflow."""
    workflow = parse_workflow_config(config)
    return run_backtest_workflow(workflow)

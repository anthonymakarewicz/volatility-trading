"""Pure orchestration service for config-driven backtest workflow runs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
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
from .workflow_types import BacktestWorkflowSpec

logger = logging.getLogger(__name__)


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

    daily_mtm = to_daily_mtm(mtm, resolved.run_config.account.initial_capital)
    report_bundle = build_backtest_report_bundle(
        trades=trades,
        mtm_daily=daily_mtm,
        run_config=_build_report_config_payload(workflow, resolved),
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


def _build_report_config_payload(
    workflow: BacktestWorkflowSpec,
    resolved: ResolvedWorkflowInputs,
) -> dict[str, Any]:
    """Build a JSON-serializable config payload for report manifests."""
    return {
        "workflow": _serialize_for_report(workflow),
        "resolved": {
            "strategy_name": resolved.strategy.name,
            "benchmark_name": resolved.benchmark_name,
            "risk_free_rate": _serialize_for_report(resolved.risk_free_rate),
        },
    }


def _serialize_for_report(value: Any) -> Any:
    """Convert workflow/runtime objects into manifest-friendly structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        first = value.index.min() if not value.empty else None
        last = value.index.max() if not value.empty else None
        return {
            "type": "series",
            "name": value.name,
            "rows": int(len(value)),
            "start": None if first is None else pd.Timestamp(first).isoformat(),
            "end": None if last is None else pd.Timestamp(last).isoformat(),
        }
    if isinstance(value, Mapping):
        return {str(key): _serialize_for_report(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_for_report(item) for item in value]
    if is_dataclass(value):
        return {
            field.name: _serialize_for_report(getattr(value, field.name))
            for field in fields(value)
        }
    return type(value).__name__

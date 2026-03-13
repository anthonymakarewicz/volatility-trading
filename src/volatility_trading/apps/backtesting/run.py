#!/usr/bin/env python
"""Run one config-driven backtest workflow."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from volatility_trading.apps._cli import (
    add_dry_run_arg,
    add_print_config_arg,
    collect_logging_overrides,
    log_dry_run,
    print_config,
)
from volatility_trading.backtesting.reporting.constants import DEFAULT_REPORT_ROOT
from volatility_trading.backtesting.runner import (
    assemble_workflow_inputs,
    parse_workflow_config,
    run_backtest_workflow_config,
)
from volatility_trading.backtesting.runner.workflow_types import BacktestWorkflowSpec
from volatility_trading.cli import (
    DEFAULT_LOGGING,
    add_config_arg,
    add_logging_args,
    build_config,
    setup_logging_from_config,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "logging": DEFAULT_LOGGING,
    "dry_run": False,
    "data": {
        "options": {
            "ticker": "SPY",
            "provider": "orats",
            "proc_root": None,
            "adapter_name": None,
            "symbol": None,
            "default_contract_multiplier": 100.0,
        },
        "rates": {
            "provider": "constant",
            "constant_rate": 0.0,
            "series_id": None,
            "column": None,
            "proc_root": None,
        },
    },
    "strategy": {
        "name": "vrp_harvesting",
        "params": {},
    },
    "account": {
        "initial_capital": 100_000.0,
    },
    "execution": {
        "option": {
            "model": "bid_ask_fee",
            "params": {},
        },
        "hedge": {
            "model": "fixed_bps",
            "params": {},
        },
    },
    "broker": {},
    "modeling": {},
    "run": {
        "start_date": None,
        "end_date": None,
    },
    "reporting": {
        "output_root": DEFAULT_REPORT_ROOT,
        "run_id": None,
        "benchmark_name": None,
        "include_dashboard_plot": True,
        "include_component_plots": False,
        "save_report_bundle": True,
    },
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the workflow runner app."""
    parser = argparse.ArgumentParser(
        description="Run one config-driven backtest workflow."
    )
    add_config_arg(parser)
    add_logging_args(parser)
    add_print_config_arg(parser)
    add_dry_run_arg(parser)

    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Override the primary underlying ticker.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Override run start date.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Override run end date.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override report output root directory.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override reporting run identifier.",
    )
    return parser.parse_args(argv)


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Build direct config-tree overrides from parsed CLI arguments."""
    overrides: dict[str, Any] = {}

    run: dict[str, Any] = {}
    if args.start is not None:
        run["start_date"] = args.start
    if args.end is not None:
        run["end_date"] = args.end
    if run:
        overrides["run"] = run

    reporting: dict[str, Any] = {}
    if args.output_root is not None:
        reporting["output_root"] = args.output_root
    if args.run_id is not None:
        reporting["run_id"] = args.run_id
    if reporting:
        overrides["reporting"] = reporting

    if args.dry_run:
        overrides["dry_run"] = True

    logging_overrides = collect_logging_overrides(args)
    if logging_overrides:
        overrides["logging"] = logging_overrides

    return overrides


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the final merged workflow config for this CLI invocation."""
    config = build_config(DEFAULT_CONFIG, args.config, _build_overrides(args))
    _apply_ticker_override(config, args.ticker)
    return config


def _workflow_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Strip app-only keys before handing config to the runner layer."""
    return {
        key: value for key, value in config.items() if key not in {"dry_run", "logging"}
    }


def _apply_ticker_override(config: dict[str, Any], ticker: str | None) -> None:
    """Apply one top-level ticker override across matching data sources."""
    if ticker is None:
        return

    data = config.get("data")
    if not isinstance(data, dict):
        return

    options = data.get("options")
    if not isinstance(options, dict):
        return

    old_ticker = _normalize_upper(options.get("ticker"))
    options["ticker"] = ticker

    features = data.get("features")
    if isinstance(features, dict):
        features_ticker = _normalize_upper(features.get("ticker"))
        if features_ticker is None or features_ticker == old_ticker:
            features["ticker"] = ticker

    hedge = data.get("hedge")
    if isinstance(hedge, dict):
        hedge_ticker = _normalize_upper(hedge.get("ticker"))
        if hedge_ticker == old_ticker:
            hedge["ticker"] = ticker
            hedge_symbol = _normalize_upper(hedge.get("symbol"))
            if hedge_symbol is None or hedge_symbol == old_ticker:
                hedge["symbol"] = ticker


def _normalize_upper(value: Any) -> str | None:
    """Normalize one optional ticker-like label."""
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized.upper()


def _build_dry_run_plan(
    workflow: BacktestWorkflowSpec,
    resolved: Any,
) -> dict[str, Any]:
    """Build one log-friendly dry-run plan summary."""
    return {
        "action": "backtest_run",
        "strategy": {
            "name": workflow.strategy.name,
            "params": dict(workflow.strategy.params),
            "signal": (
                None
                if workflow.strategy.signal is None
                else {
                    "name": workflow.strategy.signal.name,
                    "params": dict(workflow.strategy.signal.params),
                }
            ),
        },
        "data": {
            "options": {
                "ticker": workflow.data.options.ticker,
                "provider": workflow.data.options.provider,
                "proc_root": workflow.data.options.proc_root,
                "adapter_name": workflow.data.options.adapter_name,
            },
            "features": (
                None
                if workflow.data.features is None
                else {
                    "ticker": workflow.data.features.ticker,
                    "provider": workflow.data.features.provider,
                    "proc_root": workflow.data.features.proc_root,
                }
            ),
            "hedge": (
                None
                if workflow.data.hedge is None
                else {
                    "ticker": workflow.data.hedge.ticker,
                    "provider": workflow.data.hedge.provider,
                    "proc_root": workflow.data.hedge.proc_root,
                    "price_column": workflow.data.hedge.price_column,
                }
            ),
            "benchmark": (
                None
                if workflow.data.benchmark is None
                else {
                    "ticker": workflow.data.benchmark.ticker,
                    "provider": workflow.data.benchmark.provider,
                    "proc_root": workflow.data.benchmark.proc_root,
                    "price_column": workflow.data.benchmark.price_column,
                }
            ),
            "rates": (
                None
                if workflow.data.rates is None
                else {
                    "provider": workflow.data.rates.provider,
                    "constant_rate": workflow.data.rates.constant_rate,
                    "series_id": workflow.data.rates.series_id,
                    "column": workflow.data.rates.column,
                    "proc_root": workflow.data.rates.proc_root,
                }
            ),
        },
        "run": {
            "start_date": _timestamp_text(workflow.run.start_date),
            "end_date": _timestamp_text(workflow.run.end_date),
        },
        "reporting": {
            "output_root": workflow.reporting.output_root,
            "run_id": workflow.reporting.run_id,
            "save_report_bundle": workflow.reporting.save_report_bundle,
            "include_dashboard_plot": workflow.reporting.include_dashboard_plot,
            "include_component_plots": workflow.reporting.include_component_plots,
        },
        "resolved": {
            "strategy_name": resolved.strategy.name,
            "benchmark_name": resolved.benchmark_name,
            "risk_free_rate": _summarize_rate_input(resolved.risk_free_rate),
        },
    }


def _timestamp_text(value: pd.Timestamp | str | None) -> str | None:
    """Convert one optional timestamp to ISO text."""
    if value is None:
        return None
    return pd.Timestamp(value).isoformat()


def _summarize_rate_input(value: Any) -> Any:
    """Summarize one resolved risk-free-rate input for dry-run logs."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Series):
        start = None if value.empty else pd.Timestamp(value.index.min()).isoformat()
        end = None if value.empty else pd.Timestamp(value.index.max()).isoformat()
        return {
            "type": "series",
            "name": value.name,
            "rows": int(len(value)),
            "start": start,
            "end": end,
        }
    return type(value).__name__


def main(argv: list[str] | None = None) -> None:
    """Run the config-driven backtest workflow entrypoint."""
    args = _parse_args(argv)
    config = _build_config(args)
    workflow_config = _workflow_config_payload(config)

    if args.print_config:
        print_config(config)
        return

    setup_logging_from_config(config.get("logging"))
    dry_run = bool(config.get("dry_run", False))

    if dry_run:
        workflow = parse_workflow_config(workflow_config)
        resolved = assemble_workflow_inputs(workflow)
        log_dry_run(logger, _build_dry_run_plan(workflow, resolved))
        return

    result = run_backtest_workflow_config(workflow_config)
    logger.info(
        "Completed backtest workflow strategy=%s trades=%d mtm_rows=%d report_dir=%s",
        result.workflow.strategy.name,
        len(result.trades),
        len(result.mtm),
        result.report_dir,
    )


if __name__ == "__main__":
    main()

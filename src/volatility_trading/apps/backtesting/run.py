#!/usr/bin/env python
"""Run one config-driven backtest workflow."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from volatility_trading.apps._cli import (
    add_dry_run_arg,
    add_print_config_arg,
    collect_logging_overrides,
    log_dry_run,
    print_config,
)
from volatility_trading.backtesting.runner.assembly import assemble_workflow_inputs
from volatility_trading.backtesting.runner.config_parser import parse_workflow_config
from volatility_trading.backtesting.runner.defaults import (
    DEFAULT_RUNNER_APP_CONFIG,
    RUNNER_APP_ONLY_CONFIG_KEYS,
)
from volatility_trading.backtesting.runner.serialization import build_dry_run_plan
from volatility_trading.backtesting.runner.service import run_backtest_workflow_config
from volatility_trading.cli import (
    add_config_arg,
    add_logging_args,
    build_config,
    setup_logging_from_config,
)

logger = logging.getLogger(__name__)


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
    config = build_config(
        DEFAULT_RUNNER_APP_CONFIG, args.config, _build_overrides(args)
    )
    _apply_ticker_override(config, args.ticker)
    return config


def _workflow_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Strip app-only keys before handing config to the runner layer."""
    return {
        key: value
        for key, value in config.items()
        if key not in RUNNER_APP_ONLY_CONFIG_KEYS
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
        log_dry_run(logger, build_dry_run_plan(workflow, resolved))
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

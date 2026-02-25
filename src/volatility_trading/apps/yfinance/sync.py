#!/usr/bin/env python
"""Sync yfinance OHLCV time-series data to raw and processed parquet."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from volatility_trading.apps._cli import (
    add_dry_run_arg,
    add_print_config_arg,
    collect_logging_overrides,
    ensure_list,
    log_dry_run,
    print_config,
)
from volatility_trading.cli import (
    DEFAULT_LOGGING,
    add_config_arg,
    add_logging_args,
    build_config,
    resolve_path,
    setup_logging_from_config,
)
from volatility_trading.config.paths import (
    PROC_YFINANCE_TIME_SERIES,
    RAW_YFINANCE_TIME_SERIES,
)
from volatility_trading.etl.yfinance import sync_yfinance_time_series

DEFAULT_CONFIG: dict[str, Any] = {
    "logging": DEFAULT_LOGGING,
    "dry_run": False,
    "paths": {
        "raw_root": RAW_YFINANCE_TIME_SERIES,
        "proc_root": PROC_YFINANCE_TIME_SERIES,
    },
    "tickers": ["SPY", "QQQ", "IWM", "VIXY", "SP500TR", "VIX"],
    "start": "2007-01-01",
    "end": None,
    "interval": "1d",
    "auto_adjust": False,
    "actions": False,
    "overwrite": True,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync yfinance time-series to parquet files."
    )
    add_config_arg(parser)
    add_logging_args(parser)
    add_print_config_arg(parser)
    add_dry_run_arg(parser)

    parser.add_argument("--raw-root", type=str, default=None)
    parser.add_argument("--proc-root", type=str, default=None)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--interval", type=str, default=None)

    parser.add_argument(
        "--auto-adjust",
        dest="auto_adjust",
        action="store_true",
        help="Use auto-adjusted prices.",
    )
    parser.add_argument(
        "--no-auto-adjust",
        dest="auto_adjust",
        action="store_false",
        help="Keep unadjusted OHLC and Adj Close.",
    )
    parser.set_defaults(auto_adjust=None)

    parser.add_argument(
        "--actions",
        dest="actions",
        action="store_true",
        help="Request dividend/split action columns.",
    )
    parser.add_argument(
        "--no-actions",
        dest="actions",
        action="store_false",
        help="Skip action columns.",
    )
    parser.set_defaults(actions=None)

    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing parquet outputs.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Keep existing outputs.",
    )
    parser.set_defaults(overwrite=None)
    return parser.parse_args(argv)


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    paths: dict[str, Any] = {}

    if args.raw_root:
        paths["raw_root"] = args.raw_root
    if args.proc_root:
        paths["proc_root"] = args.proc_root
    if paths:
        overrides["paths"] = paths

    if args.tickers is not None:
        overrides["tickers"] = args.tickers
    if args.start is not None:
        overrides["start"] = args.start
    if args.end is not None:
        overrides["end"] = args.end
    if args.interval is not None:
        overrides["interval"] = args.interval
    if args.auto_adjust is not None:
        overrides["auto_adjust"] = args.auto_adjust
    if args.actions is not None:
        overrides["actions"] = args.actions
    if args.overwrite is not None:
        overrides["overwrite"] = args.overwrite
    if args.dry_run:
        overrides["dry_run"] = True

    logging_overrides = collect_logging_overrides(args)
    if logging_overrides:
        overrides["logging"] = logging_overrides

    return overrides


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = build_config(DEFAULT_CONFIG, args.config, _build_overrides(args))
    if args.print_config:
        print_config(config)
        return

    setup_logging_from_config(config.get("logging"))
    logger = logging.getLogger(__name__)

    raw_root = resolve_path(config["paths"]["raw_root"])
    proc_root = resolve_path(config["paths"]["proc_root"])
    if raw_root is None or proc_root is None:
        raise ValueError("paths.raw_root and paths.proc_root must be set.")

    tickers = ensure_list(config.get("tickers")) or []
    if not tickers:
        raise ValueError("tickers must not be empty.")

    start = config.get("start")
    end = config.get("end")
    interval = config.get("interval", "1d")
    auto_adjust = bool(config.get("auto_adjust", False))
    actions = bool(config.get("actions", False))
    overwrite = bool(config.get("overwrite", False))
    dry_run = bool(config.get("dry_run", False))

    logger.info("RAW root:    %s", raw_root)
    logger.info("PROC root:   %s", proc_root)
    logger.info("Tickers:     %s", tickers)
    logger.info("Window:      %s -> %s", start, end)
    logger.info("Interval:    %s", interval)
    logger.info("Auto adjust: %s", auto_adjust)
    logger.info("Actions:     %s", actions)
    logger.info("Overwrite:   %s", overwrite)

    if dry_run:
        log_dry_run(
            logger,
            {
                "action": "yfinance_sync",
                "raw_root": raw_root,
                "proc_root": proc_root,
                "tickers": tickers,
                "start": start,
                "end": end,
                "interval": interval,
                "auto_adjust": auto_adjust,
                "actions": actions,
                "overwrite": overwrite,
            },
        )
        return

    out_path = sync_yfinance_time_series(
        tickers=tickers,
        raw_root=raw_root,
        proc_root=proc_root,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        actions=actions,
        overwrite=overwrite,
    )
    logger.info("Processed file: %s", out_path)


if __name__ == "__main__":
    main()

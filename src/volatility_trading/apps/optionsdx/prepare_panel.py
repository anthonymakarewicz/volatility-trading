#!/usr/bin/env python
"""Build ticker-partitioned processed OptionsDX panels from raw `.7z` archives."""

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
from volatility_trading.config.paths import PROC_OPTIONSDX, RAW_OPTIONSDX
from volatility_trading.etl.optionsdx import prepare_optionsdx_panels

DEFAULT_CONFIG: dict[str, Any] = {
    "logging": DEFAULT_LOGGING,
    "dry_run": False,
    "paths": {
        "raw_root": RAW_OPTIONSDX,
        "proc_root": PROC_OPTIONSDX,
    },
    "tickers": ["SPY"],
    "start_year": 2010,
    "end_year": 2023,
    "low_memory": False,
    "do_clean": True,
    "reshape": "wide",
    "overwrite": True,
    "verbose": True,
    "panel_name": None,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed OptionsDX panels from raw archives."
    )
    add_config_arg(parser)
    add_logging_args(parser)
    add_print_config_arg(parser)
    add_dry_run_arg(parser)

    parser.add_argument("--raw-root", type=str, default=None)
    parser.add_argument("--proc-root", type=str, default=None)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument(
        "--all-tickers",
        action="store_true",
        help="Auto-discover all ticker subdirectories under raw root.",
    )
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--panel-name", type=str, default=None)
    parser.add_argument(
        "--reshape",
        choices=["wide", "long", "none"],
        default=None,
        help="Output shape: wide, long, or none.",
    )
    parser.add_argument(
        "--low-memory",
        dest="low_memory",
        action="store_true",
        help="Enable low-memory CSV parsing mode.",
    )
    parser.add_argument(
        "--no-low-memory",
        dest="low_memory",
        action="store_false",
        help="Disable low-memory CSV parsing mode.",
    )
    parser.set_defaults(low_memory=None)
    parser.add_argument(
        "--clean",
        dest="do_clean",
        action="store_true",
        help="Apply data cleaning.",
    )
    parser.add_argument(
        "--no-clean",
        dest="do_clean",
        action="store_false",
        help="Skip data cleaning.",
    )
    parser.set_defaults(do_clean=None)
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip existing outputs.",
    )
    parser.set_defaults(overwrite=None)
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose panel preparation logs.",
    )
    parser.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Disable verbose panel preparation logs.",
    )
    parser.set_defaults(verbose=None)

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

    if args.all_tickers:
        overrides["tickers"] = None
    elif args.tickers is not None:
        overrides["tickers"] = args.tickers

    if args.start_year is not None:
        overrides["start_year"] = args.start_year
    if args.end_year is not None:
        overrides["end_year"] = args.end_year
    if args.panel_name is not None:
        overrides["panel_name"] = args.panel_name
    if args.reshape is not None:
        overrides["reshape"] = None if args.reshape == "none" else args.reshape
    if args.low_memory is not None:
        overrides["low_memory"] = args.low_memory
    if args.do_clean is not None:
        overrides["do_clean"] = args.do_clean
    if args.overwrite is not None:
        overrides["overwrite"] = args.overwrite
    if args.verbose is not None:
        overrides["verbose"] = args.verbose
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

    tickers = ensure_list(config.get("tickers"))
    start_year = int(config.get("start_year", 2010))
    end_year = int(config.get("end_year", 2023))
    low_memory = bool(config.get("low_memory", False))
    do_clean = bool(config.get("do_clean", True))
    reshape = config.get("reshape", "wide")
    overwrite = bool(config.get("overwrite", True))
    verbose = bool(config.get("verbose", True))
    panel_name = config.get("panel_name")
    dry_run = bool(config.get("dry_run", False))

    logger.info("RAW root:     %s", raw_root)
    logger.info("PROC root:    %s", proc_root)
    logger.info("Tickers:      %s", tickers if tickers is not None else "AUTO")
    logger.info("Years:        %s -> %s", start_year, end_year)
    logger.info("Reshape:      %s", reshape)
    logger.info("Clean:        %s", do_clean)
    logger.info("Low memory:   %s", low_memory)
    logger.info("Overwrite:    %s", overwrite)
    logger.info("Panel name:   %s", panel_name if panel_name else "AUTO")

    if dry_run:
        log_dry_run(
            logger,
            {
                "action": "optionsdx_prepare_panel",
                "raw_root": raw_root,
                "proc_root": proc_root,
                "tickers": tickers,
                "start_year": start_year,
                "end_year": end_year,
                "reshape": reshape,
                "do_clean": do_clean,
                "low_memory": low_memory,
                "overwrite": overwrite,
                "panel_name": panel_name,
            },
        )
        return

    out_paths = prepare_optionsdx_panels(
        raw_root=raw_root,
        proc_root=proc_root,
        tickers=tickers,
        start_year=start_year,
        end_year=end_year,
        low_memory=low_memory,
        do_clean=do_clean,
        reshape=reshape,
        verbose=verbose,
        overwrite=overwrite,
        panel_name=panel_name,
    )
    for ticker, path in sorted(out_paths.items()):
        logger.info("Processed %s -> %s", ticker, path)


if __name__ == "__main__":
    main()

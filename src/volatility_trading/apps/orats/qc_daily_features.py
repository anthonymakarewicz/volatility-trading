#!/usr/bin/env python
"""Run QC on processed ORATS daily-features panels.

Typical usage:
    python -m volatility_trading.apps.orats.qc_daily_features --config config/orats_qc_daily_features.yml
    python -m volatility_trading.apps.orats.qc_daily_features --tickers SPX AAPL
    orats-qc-daily-features --config config/orats_qc_daily_features.yml

Config precedence: CLI > YAML > defaults.
"""

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
from volatility_trading.config.paths import PROC_ORATS_DAILY_FEATURES
from volatility_trading.etl.orats.qc.api import run_daily_features_qc

DEFAULT_CONFIG: dict[str, Any] = {
    "logging": DEFAULT_LOGGING,
    "dry_run": False,
    "paths": {
        "proc_root": PROC_ORATS_DAILY_FEATURES,
    },
    "tickers": ["AAPL"],
    "write_json": True,
    "out_json": None,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the daily-features QC app."""
    parser = argparse.ArgumentParser(
        description="Run QC on ORATS daily-features panels."
    )
    add_config_arg(parser)
    add_logging_args(parser)
    add_print_config_arg(parser)
    add_dry_run_arg(parser)

    parser.add_argument(
        "--proc-root",
        type=str,
        default=None,
        help="Processed daily-features root.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Underlying tickers to QC (space-separated).",
    )
    parser.add_argument(
        "--write-json",
        dest="write_json",
        action="store_true",
        help="Write qc_summary.json and qc_config.json.",
    )
    parser.add_argument(
        "--no-write-json",
        dest="write_json",
        action="store_false",
        help="Do not write JSON reports.",
    )
    parser.set_defaults(write_json=None)
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Explicit output path for qc_summary.json (single ticker only).",
    )

    return parser.parse_args(argv)


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Build config overrides from parsed CLI arguments."""
    overrides: dict[str, Any] = {}

    paths: dict[str, Any] = {}
    if args.proc_root:
        paths["proc_root"] = args.proc_root
    if paths:
        overrides["paths"] = paths

    if args.tickers is not None:
        overrides["tickers"] = args.tickers

    if args.write_json is not None:
        overrides["write_json"] = args.write_json
    if args.out_json is not None:
        overrides["out_json"] = args.out_json

    if args.dry_run:
        overrides["dry_run"] = True

    logging_overrides = collect_logging_overrides(args)
    if logging_overrides:
        overrides["logging"] = logging_overrides

    return overrides


def main(argv: list[str] | None = None) -> None:
    """Run the daily-features QC entrypoint."""
    args = _parse_args(argv)
    overrides = _build_overrides(args)
    config = build_config(DEFAULT_CONFIG, args.config, overrides)

    if args.print_config:
        print_config(config)
        return

    setup_logging_from_config(config.get("logging"))
    logger = logging.getLogger(__name__)

    proc_root = resolve_path(config["paths"]["proc_root"])
    if proc_root is None:
        raise ValueError("proc_root must be set.")

    tickers = ensure_list(config.get("tickers"))
    if not tickers:
        raise ValueError("tickers must be set.")

    write_json = config["write_json"]
    out_json = config.get("out_json")
    if out_json is not None and len(tickers) > 1:
        raise ValueError("--out-json requires a single ticker.")
    dry_run = config.get("dry_run", False)

    logger.info("PROC root:  %s", proc_root)
    logger.info("Tickers:    %s", tickers)
    logger.info("Write JSON: %s", write_json)
    if out_json:
        logger.info("Out JSON:   %s", out_json)

    if dry_run:
        log_dry_run(
            logger,
            {
                "action": "orats_qc_daily_features",
                "proc_root": proc_root,
                "tickers": tickers,
                "write_json": write_json,
                "out_json": out_json,
            },
        )
        return

    all_passed = True
    for ticker in tickers:
        logger.info("Running daily-features QC ticker=%s", ticker)
        result = run_daily_features_qc(
            ticker=ticker,
            proc_root=proc_root,
            out_json=out_json,
            write_json=write_json,
        )
        if not result.passed:
            all_passed = False

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

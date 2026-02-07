#!/usr/bin/env python
"""Run QC on processed ORATS daily-features panels.

Typical usage:
    python scripts/qc_orats_daily_features.py --config config/orats_qc_daily_features.yml
    python scripts/qc_orats_daily_features.py --tickers SPX AAPL

Config precedence: CLI > YAML > defaults.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

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
    "paths": {
        "proc_root": PROC_ORATS_DAILY_FEATURES,
    },
    "tickers": ["AAPL"],
    "write_json": True,
    "out_json": None,
}


def _ensure_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QC on ORATS daily-features panels."
    )
    add_config_arg(parser)
    add_logging_args(parser)

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

    return parser.parse_args()


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
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

    logging_overrides: dict[str, Any] = {}
    if args.log_level:
        logging_overrides["level"] = args.log_level
    if args.log_file:
        logging_overrides["file"] = args.log_file
    if args.log_format:
        logging_overrides["format"] = args.log_format
    if args.log_color is not None:
        logging_overrides["color"] = args.log_color
    if logging_overrides:
        overrides["logging"] = logging_overrides

    return overrides


def main() -> None:
    args = _parse_args()
    overrides = _build_overrides(args)
    config = build_config(DEFAULT_CONFIG, args.config, overrides)

    setup_logging_from_config(config.get("logging"))
    logger = logging.getLogger(__name__)

    proc_root = resolve_path(config["paths"]["proc_root"])
    if proc_root is None:
        raise ValueError("proc_root must be set.")

    tickers = _ensure_list(config.get("tickers"))
    if not tickers:
        raise ValueError("tickers must be set.")

    write_json = config["write_json"]
    out_json = config.get("out_json")
    if out_json is not None and len(tickers) > 1:
        raise ValueError("--out-json requires a single ticker.")

    logger.info("PROC root:  %s", proc_root)
    logger.info("Tickers:    %s", tickers)
    logger.info("Write JSON: %s", write_json)
    if out_json:
        logger.info("Out JSON:   %s", out_json)

    for ticker in tickers:
        logger.info("Running daily-features QC ticker=%s", ticker)
        run_daily_features_qc(
            ticker=ticker,
            proc_root=proc_root,
            out_json=out_json,
            write_json=write_json,
        )


if __name__ == "__main__":
    main()
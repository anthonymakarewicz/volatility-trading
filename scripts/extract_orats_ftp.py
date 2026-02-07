#!/usr/bin/env python
"""Extract selected underlyings from raw ORATS FTP ZIP files to Parquet.

Typical usage:
    python scripts/extract_orats_ftp.py --config config/orats_ftp_extract.yml
    python scripts/extract_orats_ftp.py --tickers SPX SPY --years 2019 2020

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
from volatility_trading.config.paths import INTER_ORATS_FTP, RAW_ORATS_FTP
from volatility_trading.etl.orats.ftp import extract


DEFAULT_CONFIG: dict[str, Any] = {
    "logging": DEFAULT_LOGGING,
    "paths": {
        "raw_root": RAW_ORATS_FTP,
        "out_root": INTER_ORATS_FTP,
    },
    "tickers": ["SPX"],
    "year_whitelist": [2007],
    "strict": True,
}


def _ensure_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ORATS FTP ZIPs into intermediate Parquet."
    )
    add_config_arg(parser)
    add_logging_args(parser)

    parser.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help="Raw ORATS FTP root directory.",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Output directory for intermediate Parquet.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Underlying tickers to extract (space-separated).",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Year allowlist (space-separated).",
    )
    parser.add_argument(
        "--all-years",
        action="store_true",
        help="Do not restrict by year.",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Raise if any ZIP files fail.",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Continue even if some ZIP files fail.",
    )
    parser.set_defaults(strict=None)

    return parser.parse_args()


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    paths: dict[str, Any] = {}
    if args.raw_root:
        paths["raw_root"] = args.raw_root
    if args.out_root:
        paths["out_root"] = args.out_root
    if paths:
        overrides["paths"] = paths

    if args.tickers is not None:
        overrides["tickers"] = args.tickers

    if args.all_years:
        overrides["year_whitelist"] = None
    elif args.years is not None:
        overrides["year_whitelist"] = args.years

    if args.strict is not None:
        overrides["strict"] = args.strict

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

    raw_root = resolve_path(config["paths"]["raw_root"])
    out_root = resolve_path(config["paths"]["out_root"])
    if raw_root is None or out_root is None:
        raise ValueError("Both raw_root and out_root must be set.")

    tickers = _ensure_list(config.get("tickers"))
    if not tickers:
        raise ValueError("tickers must be set for FTP extraction.")

    year_whitelist = _ensure_list(config.get("year_whitelist"))
    strict = config["strict"]

    logger.info("Raw ORATS root:          %s", raw_root)
    logger.info("Output (by-ticker) root: %s", out_root)
    logger.info("Tickers:                 %s", tickers)
    logger.info(
        "Years:                   %s",
        "ALL" if year_whitelist is None else sorted(str(y) for y in year_whitelist),
    )
    logger.info("Strict:                  %s", strict)

    out_root.mkdir(parents=True, exist_ok=True)

    extract(
        raw_root=raw_root,
        out_root=out_root,
        tickers=tickers,
        year_whitelist=year_whitelist,
        strict=strict,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Extract selected underlyings from raw ORATS FTP ZIP files to Parquet.

Typical usage:
    python -m volatility_trading.apps.orats.extract_ftp --config config/orats_ftp_extract.yml
    python -m volatility_trading.apps.orats.extract_ftp --tickers SPX SPY --years 2019 2020
    orats-ftp-extract --config config/orats_ftp_extract.yml

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
from volatility_trading.config.paths import INTER_ORATS_FTP, RAW_ORATS_FTP
from volatility_trading.etl.orats.ftp import extract

DEFAULT_CONFIG: dict[str, Any] = {
    "logging": DEFAULT_LOGGING,
    "dry_run": False,
    "paths": {
        "raw_root": RAW_ORATS_FTP,
        "inter_root": INTER_ORATS_FTP,
    },
    "tickers": ["SPX"],
    "year_whitelist": list(range(2007, 2026)),
    "strict": True,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the ORATS FTP extract app."""
    parser = argparse.ArgumentParser(
        description="Extract ORATS FTP ZIPs into intermediate Parquet."
    )
    add_config_arg(parser)
    add_logging_args(parser)
    add_print_config_arg(parser)
    add_dry_run_arg(parser)

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

    return parser.parse_args(argv)


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Build config overrides from parsed CLI arguments."""
    overrides: dict[str, Any] = {}

    paths: dict[str, Any] = {}
    if args.raw_root:
        paths["raw_root"] = args.raw_root
    if args.out_root:
        paths["inter_root"] = args.out_root
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

    if args.dry_run:
        overrides["dry_run"] = True

    logging_overrides = collect_logging_overrides(args)
    if logging_overrides:
        overrides["logging"] = logging_overrides

    return overrides


def main(argv: list[str] | None = None) -> None:
    """Run the ORATS FTP extract entrypoint."""
    args = _parse_args(argv)
    overrides = _build_overrides(args)
    config = build_config(DEFAULT_CONFIG, args.config, overrides)
    if args.print_config:
        print_config(config)
        return

    setup_logging_from_config(config.get("logging"))
    logger = logging.getLogger(__name__)

    raw_root = resolve_path(config["paths"]["raw_root"])
    out_root = resolve_path(config["paths"]["inter_root"])
    if raw_root is None or out_root is None:
        raise ValueError("Both raw_root and inter_root must be set.")

    tickers = ensure_list(config.get("tickers"))
    if not tickers:
        raise ValueError("tickers must be set for FTP extraction.")

    year_whitelist = ensure_list(config.get("year_whitelist"))
    strict = config["strict"]
    dry_run = config.get("dry_run", False)

    logger.info("Raw ORATS root:          %s", raw_root)
    logger.info("Output (by-ticker) root: %s", out_root)
    logger.info("Tickers:                 %s", tickers)
    logger.info(
        "Years:                   %s",
        "ALL" if year_whitelist is None else sorted(str(y) for y in year_whitelist),
    )
    logger.info("Strict:                  %s", strict)

    if dry_run:
        log_dry_run(
            logger,
            {
                "action": "orats_ftp_extract",
                "raw_root": raw_root,
                "inter_root": out_root,
                "tickers": tickers,
                "year_whitelist": year_whitelist,
                "strict": strict,
            },
        )
        return

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

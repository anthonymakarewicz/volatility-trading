#!/usr/bin/env python
"""Extract ORATS API raw snapshots into intermediate parquet.

Typical usage
-------------
    python -m volatility_trading.apps.orats.extract_api --config config/orats_api_extract.yml
    python -m volatility_trading.apps.orats.extract_api --endpoint hvs --tickers SPX NDX
    orats-api-extract --config config/orats_api_extract.yml

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
from volatility_trading.config.paths import INTER_ORATS_API, RAW_ORATS_API
from volatility_trading.etl.orats.api import extract

DEFAULT_CONFIG: dict[str, Any] = {
    "logging": DEFAULT_LOGGING,
    "dry_run": False,
    "paths": {
        "raw_root": RAW_ORATS_API,
        "inter_root": INTER_ORATS_API,
    },
    "endpoint": "hvs",
    "tickers": [
        "SPX",
        "NDX",
        "VIX",
        "SPY",
        "QQQ",
        "IWM",
        "AAPL",
        "TSLA",
        "NVDA",
        "MSFT",
    ],
    "year_whitelist": list(range(2007, 2026)),
    "raw_compression": "gz",
    "overwrite": True,
    "parquet_compression": "zstd",
    # If True, exit non-zero if any raw files fail to read/parse.
    "fail_on_failed": True,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the ORATS API extract app."""
    parser = argparse.ArgumentParser(
        description="Extract ORATS API snapshots into intermediate parquet."
    )
    add_config_arg(parser)
    add_logging_args(parser)
    add_print_config_arg(parser)
    add_dry_run_arg(parser)

    parser.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help="Raw ORATS API root directory.",
    )
    parser.add_argument(
        "--intermediate-root",
        type=str,
        default=None,
        help="Intermediate ORATS API root directory.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Endpoint name (must exist in ORATS endpoints mapping).",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Ticker allowlist (space-separated).",
    )
    parser.add_argument(
        "--all-tickers",
        action="store_true",
        help="Ignore ticker allowlist and use all tickers in data.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Year allowlist for BY_TRADE_DATE endpoints.",
    )
    parser.add_argument(
        "--all-years",
        action="store_true",
        help="Do not restrict by year (FULL_HISTORY endpoints only).",
    )
    parser.add_argument(
        "--raw-compression",
        type=str,
        default=None,
        choices=["gz", "none"],
        help="Raw snapshot compression ('gz' or 'none').",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing intermediate files.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip files that already exist.",
    )
    parser.set_defaults(overwrite=None)
    parser.add_argument(
        "--parquet-compression",
        type=str,
        default=None,
        help="Parquet compression codec (e.g., zstd, snappy).",
    )

    parser.add_argument(
        "--fail-on-failed",
        dest="fail_on_failed",
        action="store_true",
        help="Exit non-zero if any raw files fail during extraction.",
    )
    parser.add_argument(
        "--no-fail-on-failed",
        dest="fail_on_failed",
        action="store_false",
        help="Do not fail the exit code if some raw files fail.",
    )
    parser.set_defaults(fail_on_failed=None)

    return parser.parse_args(argv)


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Build config overrides from parsed CLI arguments."""
    overrides: dict[str, Any] = {}

    paths: dict[str, Any] = {}
    if args.raw_root:
        paths["raw_root"] = args.raw_root
    if args.intermediate_root:
        paths["inter_root"] = args.intermediate_root
    if paths:
        overrides["paths"] = paths

    if args.endpoint:
        overrides["endpoint"] = args.endpoint

    if args.all_tickers:
        overrides["tickers"] = None
    elif args.tickers is not None:
        overrides["tickers"] = args.tickers

    if args.all_years:
        overrides["year_whitelist"] = None
    elif args.years is not None:
        overrides["year_whitelist"] = args.years

    if args.raw_compression:
        overrides["raw_compression"] = args.raw_compression
    if args.overwrite is not None:
        overrides["overwrite"] = args.overwrite
    if args.parquet_compression:
        overrides["parquet_compression"] = args.parquet_compression

    if args.fail_on_failed is not None:
        overrides["fail_on_failed"] = args.fail_on_failed

    if args.dry_run:
        overrides["dry_run"] = True

    logging_overrides = collect_logging_overrides(args)
    if logging_overrides:
        overrides["logging"] = logging_overrides

    return overrides


def main(argv: list[str] | None = None) -> None:
    """Run the ORATS API extract entrypoint."""
    args = _parse_args(argv)
    overrides = _build_overrides(args)
    config = build_config(DEFAULT_CONFIG, args.config, overrides)
    if args.print_config:
        print_config(config)
        return

    setup_logging_from_config(config.get("logging"))
    logger = logging.getLogger(__name__)

    raw_root = resolve_path(config["paths"]["raw_root"])
    inter_root = resolve_path(config["paths"]["inter_root"])
    if raw_root is None or inter_root is None:
        raise ValueError("Both raw_root and inter_root must be set.")

    endpoint = config["endpoint"]
    tickers = ensure_list(config.get("tickers"))
    year_whitelist = ensure_list(config.get("year_whitelist"))
    raw_compression = config["raw_compression"]
    overwrite = config["overwrite"]
    parquet_compression = config["parquet_compression"]
    dry_run = config.get("dry_run", False)

    logger.info("RAW API root:         %s", raw_root)
    logger.info("Intermediate API root:%s", inter_root)
    logger.info("Endpoint:             %s", endpoint)
    logger.info("Tickers:              %s", tickers)
    logger.info("Years:                %s", year_whitelist)
    logger.info("Raw compression:      %s", raw_compression)
    logger.info("Overwrite:            %s", overwrite)
    logger.info("Parquet compression:  %s", parquet_compression)
    logger.info("Fail on failed:       %s", config["fail_on_failed"])

    if dry_run:
        log_dry_run(
            logger,
            {
                "action": "orats_api_extract",
                "raw_root": raw_root,
                "inter_root": inter_root,
                "endpoint": endpoint,
                "tickers": tickers,
                "year_whitelist": year_whitelist,
                "raw_compression": raw_compression,
                "overwrite": overwrite,
                "parquet_compression": parquet_compression,
                "fail_on_failed": config["fail_on_failed"],
            },
        )
        return

    raw_root.mkdir(parents=True, exist_ok=True)
    inter_root.mkdir(parents=True, exist_ok=True)

    result = extract(
        endpoint=endpoint,
        raw_root=raw_root,
        intermediate_root=inter_root,
        tickers=tickers,
        year_whitelist=year_whitelist,
        compression=raw_compression,
        overwrite=overwrite,
        parquet_compression=parquet_compression,
    )

    if config["fail_on_failed"] and result.n_failed:
        logger.error(
            "Extraction finished with failures: n_failed=%d (see result.failed_paths)",
            result.n_failed,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()

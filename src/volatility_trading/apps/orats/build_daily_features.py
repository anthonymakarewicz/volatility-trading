#!/usr/bin/env python
"""Build processed daily-features panels from intermediate ORATS API data.

Typical usage:
    python -m volatility_trading.apps.orats.build_daily_features --config config/orats_daily_features_build.yml
    python -m volatility_trading.apps.orats.build_daily_features --tickers SPX --endpoints summaries hvs
    orats-build-daily-features --config config/orats_daily_features_build.yml

Config precedence: CLI > YAML > defaults.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

from volatility_trading.apps._cli import (
    add_print_config_arg,
    collect_logging_overrides,
    ensure_list,
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
    INTER_ORATS_API,
    PROC_ORATS_DAILY_FEATURES,
)
from volatility_trading.etl.orats.processed.daily_features import build


DEFAULT_CONFIG: dict[str, Any] = {
    "logging": DEFAULT_LOGGING,
    "paths": {
        "inter_api_root": INTER_ORATS_API,
        "proc_root": PROC_ORATS_DAILY_FEATURES,
    },
    "tickers": ["AAPL"],
    "endpoints": ["summaries", "hvs"],
    "prefix_endpoint_cols": True,
    "priority_endpoints": None,
    "collect_stats": True,
    "columns": None,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed ORATS daily-features panels."
    )
    add_config_arg(parser)
    add_logging_args(parser)
    add_print_config_arg(parser)

    parser.add_argument(
        "--inter-api-root",
        type=str,
        default=None,
        help="Intermediate ORATS API root directory.",
    )
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
        help="Underlying tickers to build (space-separated).",
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=None,
        help="Endpoints to include (space-separated).",
    )
    parser.add_argument(
        "--priority-endpoints",
        nargs="+",
        default=None,
        help="Endpoints to prioritize when canonicalizing columns.",
    )
    parser.add_argument(
        "--prefix-endpoint-cols",
        dest="prefix_endpoint_cols",
        action="store_true",
        help="Prefix endpoint columns before canonicalization.",
    )
    parser.add_argument(
        "--no-prefix-endpoint-cols",
        dest="prefix_endpoint_cols",
        action="store_false",
        help="Do not prefix endpoint columns.",
    )
    parser.set_defaults(prefix_endpoint_cols=None)
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Explicit output columns list.",
    )
    parser.add_argument(
        "--collect-stats",
        dest="collect_stats",
        action="store_true",
        help="Collect build stats.",
    )
    parser.add_argument(
        "--no-collect-stats",
        dest="collect_stats",
        action="store_false",
        help="Disable build stats collection.",
    )
    parser.set_defaults(collect_stats=None)

    return parser.parse_args(argv)


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    paths: dict[str, Any] = {}
    if args.inter_api_root:
        paths["inter_api_root"] = args.inter_api_root
    if args.proc_root:
        paths["proc_root"] = args.proc_root
    if paths:
        overrides["paths"] = paths

    if args.tickers is not None:
        overrides["tickers"] = args.tickers

    if args.endpoints is not None:
        overrides["endpoints"] = args.endpoints

    if args.priority_endpoints is not None:
        overrides["priority_endpoints"] = args.priority_endpoints

    if args.prefix_endpoint_cols is not None:
        overrides["prefix_endpoint_cols"] = args.prefix_endpoint_cols

    if args.columns is not None:
        overrides["columns"] = args.columns

    if args.collect_stats is not None:
        overrides["collect_stats"] = args.collect_stats

    logging_overrides = collect_logging_overrides(args)
    if logging_overrides:
        overrides["logging"] = logging_overrides

    return overrides


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    overrides = _build_overrides(args)
    config = build_config(DEFAULT_CONFIG, args.config, overrides)

    if args.print_config:
        print_config(config)
        return

    setup_logging_from_config(config.get("logging"))
    logger = logging.getLogger(__name__)

    inter_api_root = resolve_path(config["paths"]["inter_api_root"])
    proc_root = resolve_path(config["paths"]["proc_root"])
    if inter_api_root is None or proc_root is None:
        raise ValueError("inter_api_root and proc_root must be set.")

    tickers = ensure_list(config.get("tickers"))
    if not tickers:
        raise ValueError("tickers must be set.")

    endpoints = ensure_list(config.get("endpoints"))
    priority_endpoints = ensure_list(config.get("priority_endpoints"))
    prefix_endpoint_cols = config["prefix_endpoint_cols"]
    collect_stats = config["collect_stats"]
    columns = config.get("columns")

    logger.info("INTER API root:       %s", inter_api_root)
    logger.info("PROC root:            %s", proc_root)
    logger.info("Tickers:              %s", tickers)
    logger.info("Endpoints:            %s", endpoints)
    logger.info("Priority endpoints:   %s", priority_endpoints)
    logger.info("Prefix endpoint cols: %s", prefix_endpoint_cols)
    logger.info(
        "Columns:              %s",
        "DEFAULT" if columns is None else len(columns),
    )
    logger.info("Collect stats:        %s", collect_stats)

    proc_root.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        logger.info("Building daily features ticker=%s", ticker)
        build(
            inter_api_root=inter_api_root,
            proc_root=proc_root,
            ticker=ticker,
            endpoints=endpoints,
            prefix_endpoint_cols=prefix_endpoint_cols,
            priority_endpoints=priority_endpoints,
            collect_stats=collect_stats,
            columns=columns,
        )


if __name__ == "__main__":
    main()

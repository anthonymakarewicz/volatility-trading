#!/usr/bin/env python
"""Build processed options-chain panels from intermediate data.

Typical usage:
    python -m volatility_trading.apps.orats.build_options_chain --config config/orats_options_chain_build.yml
    python -m volatility_trading.apps.orats.build_options_chain --tickers SPX SPY --years 2019 2020
    orats-build-options-chain --config config/orats_options_chain_build.yml

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
from volatility_trading.config.paths import (
    INTER_ORATS_API,
    INTER_ORATS_FTP,
    PROC_ORATS_OPTIONS_CHAIN,
)
from volatility_trading.etl.orats.processed.options_chain import build


DEFAULT_CONFIG: dict[str, Any] = {
    "logging": DEFAULT_LOGGING,
    "paths": {
        "inter_strikes_root": INTER_ORATS_FTP,
        "monies_implied_inter_root": INTER_ORATS_API,
        "proc_root": PROC_ORATS_OPTIONS_CHAIN,
    },
    "tickers": ["AAPL"],
    "years": None,
    "dte_min": 1,
    "dte_max": 252,
    "moneyness_min": 0.7,
    "moneyness_max": 1.3,
    "columns": None,
    "collect_stats": True,
    "derive_put_greeks": False,
    "merge_dividend_yield": True,
}


def _ensure_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed ORATS options-chain panels."
    )
    add_config_arg(parser)
    add_logging_args(parser)

    parser.add_argument(
        "--inter-strikes-root",
        type=str,
        default=None,
        help="Intermediate ORATS FTP strikes root.",
    )
    parser.add_argument(
        "--monies-implied-root",
        type=str,
        default=None,
        help="Intermediate ORATS API root for monies_implied merge.",
    )
    parser.add_argument(
        "--proc-root",
        type=str,
        default=None,
        help="Processed options-chain root.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Underlying tickers to build (space-separated).",
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
        "--dte-min",
        type=int,
        default=None,
        help="Minimum DTE filter.",
    )
    parser.add_argument(
        "--dte-max",
        type=int,
        default=None,
        help="Maximum DTE filter.",
    )
    parser.add_argument(
        "--moneyness-min",
        type=float,
        default=None,
        help="Minimum K/S moneyness filter.",
    )
    parser.add_argument(
        "--moneyness-max",
        type=float,
        default=None,
        help="Maximum K/S moneyness filter.",
    )
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
    parser.add_argument(
        "--derive-put-greeks",
        dest="derive_put_greeks",
        action="store_true",
        help="Derive put greeks via parity.",
    )
    parser.add_argument(
        "--no-derive-put-greeks",
        dest="derive_put_greeks",
        action="store_false",
        help="Use minimal put greeks derivation.",
    )
    parser.set_defaults(derive_put_greeks=None)
    parser.add_argument(
        "--merge-dividend-yield",
        dest="merge_dividend_yield",
        action="store_true",
        help="Merge dividend_yield from monies_implied.",
    )
    parser.add_argument(
        "--no-merge-dividend-yield",
        dest="merge_dividend_yield",
        action="store_false",
        help="Disable dividend_yield merge.",
    )
    parser.set_defaults(merge_dividend_yield=None)

    return parser.parse_args(argv)


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    paths: dict[str, Any] = {}
    if args.inter_strikes_root:
        paths["inter_strikes_root"] = args.inter_strikes_root
    if args.monies_implied_root:
        paths["monies_implied_inter_root"] = args.monies_implied_root
    if args.proc_root:
        paths["proc_root"] = args.proc_root
    if paths:
        overrides["paths"] = paths

    if args.tickers is not None:
        overrides["tickers"] = args.tickers

    if args.all_years:
        overrides["years"] = None
    elif args.years is not None:
        overrides["years"] = args.years

    if args.dte_min is not None:
        overrides["dte_min"] = args.dte_min
    if args.dte_max is not None:
        overrides["dte_max"] = args.dte_max
    if args.moneyness_min is not None:
        overrides["moneyness_min"] = args.moneyness_min
    if args.moneyness_max is not None:
        overrides["moneyness_max"] = args.moneyness_max
    if args.columns is not None:
        overrides["columns"] = args.columns

    if args.collect_stats is not None:
        overrides["collect_stats"] = args.collect_stats
    if args.derive_put_greeks is not None:
        overrides["derive_put_greeks"] = args.derive_put_greeks
    if args.merge_dividend_yield is not None:
        overrides["merge_dividend_yield"] = args.merge_dividend_yield

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


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    overrides = _build_overrides(args)
    config = build_config(DEFAULT_CONFIG, args.config, overrides)

    setup_logging_from_config(config.get("logging"))
    logger = logging.getLogger(__name__)

    inter_strikes_root = resolve_path(config["paths"]["inter_strikes_root"])
    monies_implied_root = resolve_path(config["paths"]["monies_implied_inter_root"])
    proc_root = resolve_path(config["paths"]["proc_root"])
    if inter_strikes_root is None or proc_root is None:
        raise ValueError("inter_strikes_root and proc_root must be set.")

    tickers = _ensure_list(config.get("tickers"))
    if not tickers:
        raise ValueError("tickers must be set.")

    years = _ensure_list(config.get("years"))
    dte_min = config["dte_min"]
    dte_max = config["dte_max"]
    moneyness_min = config["moneyness_min"]
    moneyness_max = config["moneyness_max"]
    columns = config.get("columns")
    collect_stats = config["collect_stats"]
    derive_put_greeks = config["derive_put_greeks"]
    merge_dividend_yield = config["merge_dividend_yield"]

    logger.info("INTER strikes root: %s", inter_strikes_root)
    logger.info("INTER API root:     %s", monies_implied_root)
    logger.info("PROC root:          %s", proc_root)
    logger.info("Tickers:            %s", tickers)
    logger.info("Years:              %s", years)
    logger.info("DTE:                [%s, %s]", dte_min, dte_max)
    logger.info("Moneyness:          [%s, %s]", moneyness_min, moneyness_max)
    logger.info(
        "Columns:            %s",
        "DEFAULT" if columns is None else len(columns),
    )
    logger.info("Collect stats:      %s", collect_stats)
    logger.info("Derive put greeks:  %s", derive_put_greeks)
    logger.info("Merge dividend:     %s", merge_dividend_yield)

    proc_root.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        logger.info("Building options chain ticker=%s", ticker)
        build(
            inter_root=inter_strikes_root,
            proc_root=proc_root,
            ticker=ticker,
            years=years,
            dte_min=dte_min,
            dte_max=dte_max,
            moneyness_min=moneyness_min,
            moneyness_max=moneyness_max,
            monies_implied_inter_root=monies_implied_root,
            merge_dividend_yield=merge_dividend_yield,
            derive_put_greeks=derive_put_greeks,
            collect_stats=collect_stats,
            columns=columns,
        )


if __name__ == "__main__":
    main()

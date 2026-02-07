#!/usr/bin/env python
"""Run QC on processed ORATS options-chain panels.

Typical usage:
    python -m volatility_trading.apps.orats.qc_options_chain --config config/orats_qc_options_chain.yml
    python -m volatility_trading.apps.orats.qc_options_chain --tickers SPX AAPL
    orats-qc-options-chain --config config/orats_qc_options_chain.yml

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
from volatility_trading.config.paths import PROC_ORATS_OPTIONS_CHAIN
from volatility_trading.etl.orats.qc.api import run_options_chain_qc


DEFAULT_CONFIG: dict[str, Any] = {
    "logging": DEFAULT_LOGGING,
    "paths": {
        "proc_root": PROC_ORATS_OPTIONS_CHAIN,
    },
    "tickers": ["AAPL"],
    "write_json": True,
    "out_json": None,
    "dte_bins": [0, 10, 30, 60, 180],
    "delta_bins": [0.0, 0.05, 0.1, 0.3, 0.7, 0.9, 0.95, 1.0],
    "roi_dte_min": 10,
    "roi_dte_max": 60,
    "roi_delta_min": 0.1,
    "roi_delta_max": 0.9,
    "top_k_buckets": 10,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QC on ORATS options-chain panels."
    )
    add_config_arg(parser)
    add_logging_args(parser)
    add_print_config_arg(parser)

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
    parser.add_argument(
        "--dte-bins",
        nargs="+",
        type=float,
        default=None,
        help="DTE bin edges (space-separated).",
    )
    parser.add_argument(
        "--delta-bins",
        nargs="+",
        type=float,
        default=None,
        help="Delta bin edges (space-separated).",
    )
    parser.add_argument(
        "--roi-dte-min",
        type=int,
        default=None,
        help="ROI DTE minimum.",
    )
    parser.add_argument(
        "--roi-dte-max",
        type=int,
        default=None,
        help="ROI DTE maximum.",
    )
    parser.add_argument(
        "--roi-delta-min",
        type=float,
        default=None,
        help="ROI absolute delta minimum.",
    )
    parser.add_argument(
        "--roi-delta-max",
        type=float,
        default=None,
        help="ROI absolute delta maximum.",
    )
    parser.add_argument(
        "--top-k-buckets",
        type=int,
        default=None,
        help="Top K buckets for soft checks.",
    )

    return parser.parse_args(argv)


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

    if args.dte_bins is not None:
        overrides["dte_bins"] = args.dte_bins
    if args.delta_bins is not None:
        overrides["delta_bins"] = args.delta_bins
    if args.roi_dte_min is not None:
        overrides["roi_dte_min"] = args.roi_dte_min
    if args.roi_dte_max is not None:
        overrides["roi_dte_max"] = args.roi_dte_max
    if args.roi_delta_min is not None:
        overrides["roi_delta_min"] = args.roi_delta_min
    if args.roi_delta_max is not None:
        overrides["roi_delta_max"] = args.roi_delta_max
    if args.top_k_buckets is not None:
        overrides["top_k_buckets"] = args.top_k_buckets

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

    dte_bins = ensure_list(config.get("dte_bins")) or []
    delta_bins = ensure_list(config.get("delta_bins")) or []
    roi_dte_min = config["roi_dte_min"]
    roi_dte_max = config["roi_dte_max"]
    roi_delta_min = config["roi_delta_min"]
    roi_delta_max = config["roi_delta_max"]
    top_k_buckets = config["top_k_buckets"]

    logger.info("PROC root:      %s", proc_root)
    logger.info("Tickers:        %s", tickers)
    logger.info("Write JSON:     %s", write_json)
    if out_json:
        logger.info("Out JSON:       %s", out_json)
    logger.info("ROI DTE:        %s..%s", roi_dte_min, roi_dte_max)
    logger.info("ROI |delta|:    %s..%s", roi_delta_min, roi_delta_max)
    logger.info("DTE bins:       %s", dte_bins)
    logger.info("Delta bins:     %s", delta_bins)
    logger.info("Top K buckets:  %s", top_k_buckets)

    all_passed = True
    for ticker in tickers:
        logger.info("Running options-chain QC ticker=%s", ticker)
        result = run_options_chain_qc(
            ticker=ticker,
            proc_root=proc_root,
            out_json=out_json,
            write_json=write_json,
            dte_bins=dte_bins,
            delta_bins=delta_bins,
            roi_dte_min=roi_dte_min,
            roi_dte_max=roi_dte_max,
            roi_delta_min=roi_delta_min,
            roi_delta_max=roi_delta_max,
            top_k_buckets=top_k_buckets,
        )
        if not result.passed:
            all_passed = False

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

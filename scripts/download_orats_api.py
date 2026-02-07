#!/usr/bin/env python
"""Download ORATS API endpoint data to disk (raw).

Typical usage:
    python scripts/download_orats_api.py --config config/orats_api_download.yml
    python scripts/download_orats_api.py --endpoint ivrank --tickers SPX NDX

Notes:
- Reads ORATS token from env var ORATS_API_KEY by default.
- Config precedence: CLI > YAML > defaults.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from dotenv import load_dotenv

from volatility_trading.cli import (
    DEFAULT_LOGGING,
    add_config_arg,
    add_logging_args,
    build_config,
    resolve_path,
    setup_logging_from_config,
)
from volatility_trading.config.paths import RAW_ORATS_API
from volatility_trading.etl.orats.api import download


DEFAULT_CONFIG: dict[str, Any] = {
    "logging": DEFAULT_LOGGING,
    "api": {
        "token_env": "ORATS_API_KEY",
        "token": None,
    },
    "paths": {
        "raw_root": RAW_ORATS_API,
    },
    "endpoint": "ivrank",
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
    "year_whitelist": None,
    "fields": None,
    "compression": "gz",
    "overwrite": False,
    "sleep_s": 0.10,
}


def _ensure_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ORATS API endpoint snapshots."
    )
    add_config_arg(parser)
    add_logging_args(parser)

    parser.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help="Raw ORATS API root directory.",
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
        "--fields",
        nargs="+",
        default=None,
        help="Fields list to request (space-separated).",
    )
    parser.add_argument(
        "--all-fields",
        action="store_true",
        help="Use endpoint defaults for fields.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default=None,
        choices=["gz", "none"],
        help="Compression for raw JSON ('gz' or 'none').",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip files that already exist.",
    )
    parser.set_defaults(overwrite=None)
    parser.add_argument(
        "--sleep-s",
        type=float,
        default=None,
        help="Delay between requests (seconds).",
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="ORATS API token (overrides env).",
    )
    parser.add_argument(
        "--token-env",
        type=str,
        default=None,
        help="Env var name for ORATS API token.",
    )

    return parser.parse_args()


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    paths: dict[str, Any] = {}
    if args.raw_root:
        paths["raw_root"] = args.raw_root
    if paths:
        overrides["paths"] = paths

    if args.endpoint:
        overrides["endpoint"] = args.endpoint
    if args.tickers is not None:
        overrides["tickers"] = args.tickers

    if args.all_years:
        overrides["year_whitelist"] = None
    elif args.years is not None:
        overrides["year_whitelist"] = args.years

    if args.all_fields:
        overrides["fields"] = None
    elif args.fields is not None:
        overrides["fields"] = args.fields

    if args.compression:
        overrides["compression"] = args.compression
    if args.overwrite is not None:
        overrides["overwrite"] = args.overwrite
    if args.sleep_s is not None:
        overrides["sleep_s"] = args.sleep_s

    api_overrides: dict[str, Any] = {}
    if args.token is not None:
        api_overrides["token"] = args.token
    if args.token_env is not None:
        api_overrides["token_env"] = args.token_env
    if api_overrides:
        overrides["api"] = api_overrides

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
    if raw_root is None:
        raise ValueError("raw_root must be set.")

    api_cfg = config.get("api", {})
    token_env = api_cfg.get("token_env", "ORATS_API_KEY")

    load_dotenv()
    token = api_cfg.get("token") or os.getenv(token_env)
    if not token:
        raise RuntimeError(
            f"Missing ORATS API token. Set env var {token_env} or pass --token."
        )

    endpoint = config["endpoint"]
    tickers = _ensure_list(config.get("tickers"))
    year_whitelist = _ensure_list(config.get("year_whitelist"))
    fields = _ensure_list(config.get("fields"))
    compression = config["compression"]
    overwrite = config["overwrite"]
    sleep_s = config["sleep_s"]

    logger.info("RAW API root: %s", raw_root)
    logger.info("Endpoint:     %s", endpoint)
    logger.info("Tickers:      %s", tickers)
    logger.info("Years:        %s", year_whitelist)
    logger.info("Fields:       %s", "DEFAULT" if fields is None else len(fields))
    logger.info("Compression:  %s", compression)
    logger.info("Overwrite:    %s", overwrite)
    logger.info("Sleep (s):    %s", sleep_s)

    raw_root.mkdir(parents=True, exist_ok=True)

    download(
        token=token,
        endpoint=endpoint,
        raw_root=raw_root,
        tickers=tickers,
        year_whitelist=year_whitelist,
        fields=fields,
        sleep_s=sleep_s,
        overwrite=overwrite,
        compression=compression,
    )


if __name__ == "__main__":
    main()

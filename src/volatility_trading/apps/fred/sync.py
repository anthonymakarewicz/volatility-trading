#!/usr/bin/env python
"""Sync configured FRED series to raw and processed datasets."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
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
from volatility_trading.config.paths import PROC_FRED, RAW_FRED
from volatility_trading.etl.fred import sync_fred_domains

DEFAULT_CONFIG: dict[str, Any] = {
    "logging": DEFAULT_LOGGING,
    "dry_run": False,
    "paths": {
        "raw_root": RAW_FRED,
        "proc_root": PROC_FRED,
    },
    "fred": {
        "token_env": "FRED_API_KEY",
        "token": None,
    },
    "domains": {
        "rates": {
            "dgs3mo": "DGS3MO",
            "dgs2": "DGS2",
            "dgs10": "DGS10",
        },
        "market": {
            "vixcls": "VIXCLS",
        },
    },
    "domain_names": None,
    "start": "2005-01-01",
    "end": None,
    "asfreq_business_days": True,
    "overwrite": True,
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync FRED domains to parquet files.")
    add_config_arg(parser)
    add_logging_args(parser)
    add_print_config_arg(parser)
    add_dry_run_arg(parser)

    parser.add_argument("--raw-root", type=str, default=None)
    parser.add_argument("--proc-root", type=str, default=None)
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--token-env", type=str, default=None)
    parser.add_argument(
        "--business-days",
        dest="asfreq_business_days",
        action="store_true",
        help="Reindex output to business days and forward-fill.",
    )
    parser.add_argument(
        "--no-business-days",
        dest="asfreq_business_days",
        action="store_false",
        help="Keep native observation dates.",
    )
    parser.set_defaults(asfreq_business_days=None)
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing parquet outputs.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Keep existing outputs.",
    )
    parser.set_defaults(overwrite=None)
    return parser.parse_args(argv)


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    paths: dict[str, Any] = {}
    fred_cfg: dict[str, Any] = {}

    if args.raw_root:
        paths["raw_root"] = args.raw_root
    if args.proc_root:
        paths["proc_root"] = args.proc_root
    if paths:
        overrides["paths"] = paths

    if args.domains is not None:
        overrides["domain_names"] = args.domains
    if args.start is not None:
        overrides["start"] = args.start
    if args.end is not None:
        overrides["end"] = args.end
    if args.asfreq_business_days is not None:
        overrides["asfreq_business_days"] = args.asfreq_business_days
    if args.overwrite is not None:
        overrides["overwrite"] = args.overwrite

    if args.token is not None:
        fred_cfg["token"] = args.token
    if args.token_env is not None:
        fred_cfg["token_env"] = args.token_env
    if fred_cfg:
        overrides["fred"] = fred_cfg

    if args.dry_run:
        overrides["dry_run"] = True

    logging_overrides = collect_logging_overrides(args)
    if logging_overrides:
        overrides["logging"] = logging_overrides

    return overrides


def _resolve_domains(
    domain_map: Mapping[str, Any], selected_names: list[str] | None
) -> dict[str, dict[str, str]]:
    if selected_names is None:
        selected_names = list(domain_map.keys())
    selected: dict[str, dict[str, str]] = {}
    for name in selected_names:
        payload = domain_map.get(name)
        if payload is None:
            raise ValueError(f"Unknown domain '{name}'. Available: {list(domain_map)}")
        if not isinstance(payload, Mapping):
            raise ValueError(f"Domain '{name}' must map aliases to FRED series ids.")
        selected[name] = {
            str(alias): str(series_id) for alias, series_id in payload.items()
        }
    return selected


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

    domain_names = ensure_list(config.get("domain_names"))
    domains = _resolve_domains(config.get("domains", {}), domain_names)

    fred_cfg = config.get("fred", {})
    start = config.get("start")
    end = config.get("end")
    asfreq_business_days = bool(config.get("asfreq_business_days", True))
    overwrite = bool(config.get("overwrite", False))
    dry_run = bool(config.get("dry_run", False))

    logger.info("RAW root:   %s", raw_root)
    logger.info("PROC root:  %s", proc_root)
    logger.info("Domains:    %s", list(domains))
    logger.info("Window:     %s -> %s", start, end)
    logger.info("Biz ffill:  %s", asfreq_business_days)
    logger.info("Overwrite:  %s", overwrite)

    if dry_run:
        log_dry_run(
            logger,
            {
                "action": "fred_sync",
                "raw_root": raw_root,
                "proc_root": proc_root,
                "domains": domains,
                "start": start,
                "end": end,
                "asfreq_business_days": asfreq_business_days,
                "overwrite": overwrite,
                "token_env": fred_cfg.get("token_env", "FRED_API_KEY"),
                "token_present": bool(fred_cfg.get("token")),
            },
        )
        return

    out_paths = sync_fred_domains(
        raw_root=raw_root,
        proc_root=proc_root,
        domains=domains,
        start=start,
        end=end,
        token=fred_cfg.get("token"),
        token_env=fred_cfg.get("token_env", "FRED_API_KEY"),
        asfreq_business_days=asfreq_business_days,
        overwrite=overwrite,
    )
    for domain, path in out_paths.items():
        logger.info("Processed %s -> %s", domain, path)


if __name__ == "__main__":
    main()

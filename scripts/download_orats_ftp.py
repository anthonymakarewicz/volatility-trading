#!/usr/bin/env python
"""Download ORATS HostedFTP SMV Strikes ZIP files to disk (raw).

Typical usage:
    python scripts/download_orats_ftp.py --config config/orats_ftp_download.yml
    python scripts/download_orats_ftp.py --years 2020 2021 --max-workers 4

Notes
-----
- Reads FTP credentials from env vars (or a .env file at project root) unless
  provided via CLI/YAML.
- Host/remote base dirs default to the downloader defaults.
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
from volatility_trading.config.paths import RAW_ORATS_FTP
from volatility_trading.etl.orats.ftp import download


DEFAULT_CONFIG: dict[str, Any] = {
    "logging": DEFAULT_LOGGING,
    "ftp": {
        "user_env": "ORATS_FTP_USER",
        "pass_env": "ORATS_FTP_PASS",
        "user": None,
        "password": None,
        "host": None,
        "remote_base_dirs": None,
    },
    "paths": {
        "raw_root": RAW_ORATS_FTP,
    },
    "year_whitelist": None,
    "validate_zip": True,
    "max_workers": 3,
}


def _ensure_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ORATS FTP ZIP files into a raw directory."
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
        "--years",
        nargs="+",
        type=int,
        default=None,
        help="Year allowlist (space-separated).",
    )
    parser.add_argument(
        "--all-years",
        action="store_true",
        help="Download all available years.",
    )
    parser.add_argument(
        "--validate-zip",
        dest="validate_zip",
        action="store_true",
        help="Validate ZIP files during download.",
    )
    parser.add_argument(
        "--no-validate-zip",
        dest="validate_zip",
        action="store_false",
        help="Skip ZIP validation.",
    )
    parser.set_defaults(validate_zip=None)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of parallel worker threads.",
    )

    parser.add_argument(
        "--ftp-user",
        type=str,
        default=None,
        help="FTP username (overrides env).",
    )
    parser.add_argument(
        "--ftp-pass",
        type=str,
        default=None,
        help="FTP password (overrides env).",
    )
    parser.add_argument(
        "--ftp-user-env",
        type=str,
        default=None,
        help="Env var name for FTP username.",
    )
    parser.add_argument(
        "--ftp-pass-env",
        type=str,
        default=None,
        help="Env var name for FTP password.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="FTP host override (advanced).",
    )
    parser.add_argument(
        "--remote-base-dirs",
        nargs="+",
        default=None,
        help="FTP base directories override (advanced).",
    )

    return parser.parse_args()


def _build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    paths: dict[str, Any] = {}
    if args.raw_root:
        paths["raw_root"] = args.raw_root
    if paths:
        overrides["paths"] = paths

    if args.all_years:
        overrides["year_whitelist"] = None
    elif args.years is not None:
        overrides["year_whitelist"] = args.years

    if args.validate_zip is not None:
        overrides["validate_zip"] = args.validate_zip
    if args.max_workers is not None:
        overrides["max_workers"] = args.max_workers

    ftp_overrides: dict[str, Any] = {}
    if args.ftp_user is not None:
        ftp_overrides["user"] = args.ftp_user
    if args.ftp_pass is not None:
        ftp_overrides["password"] = args.ftp_pass
    if args.ftp_user_env is not None:
        ftp_overrides["user_env"] = args.ftp_user_env
    if args.ftp_pass_env is not None:
        ftp_overrides["pass_env"] = args.ftp_pass_env
    if args.host is not None:
        ftp_overrides["host"] = args.host
    if args.remote_base_dirs is not None:
        ftp_overrides["remote_base_dirs"] = args.remote_base_dirs
    if ftp_overrides:
        overrides["ftp"] = ftp_overrides

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

    ftp_cfg = config.get("ftp", {})
    user_env = ftp_cfg.get("user_env", "ORATS_FTP_USER")
    pass_env = ftp_cfg.get("pass_env", "ORATS_FTP_PASS")

    load_dotenv()
    user = ftp_cfg.get("user") or os.getenv(user_env)
    password = ftp_cfg.get("password") or os.getenv(pass_env)
    if not user or not password:
        raise RuntimeError(
            "Missing ORATS FTP credentials. Set env vars "
            f"{user_env} and {pass_env}, or pass --ftp-user/--ftp-pass."
        )

    year_whitelist = _ensure_list(config.get("year_whitelist"))
    validate_zip = config["validate_zip"]
    max_workers = config["max_workers"]
    host = ftp_cfg.get("host")
    remote_base_dirs = ftp_cfg.get("remote_base_dirs")

    logger.info("RAW FTP root:  %s", raw_root)
    logger.info(
        "Years:         %s",
        "ALL" if year_whitelist is None else sorted(str(y) for y in year_whitelist),
    )
    logger.info("Validate ZIP:  %s", validate_zip)
    logger.info("Max workers:   %s", max_workers)
    if host:
        logger.info("FTP host:      %s", host)
    if remote_base_dirs:
        logger.info("Remote dirs:   %s", remote_base_dirs)

    raw_root.mkdir(parents=True, exist_ok=True)

    download_kwargs: dict[str, Any] = {}
    if host:
        download_kwargs["host"] = host
    if remote_base_dirs:
        download_kwargs["remote_base_dirs"] = remote_base_dirs

    download(
        user=user,
        password=password,
        raw_root=raw_root,
        year_whitelist=year_whitelist,
        validate_zip=validate_zip,
        max_workers=max_workers,
        **download_kwargs,
    )


if __name__ == "__main__":
    main()

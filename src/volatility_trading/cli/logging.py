from __future__ import annotations

from typing import Any, Mapping

from volatility_trading.utils.logging_config import setup_logging


DEFAULT_LOGGING: dict[str, Any] = {
    "level": "INFO",
    "format": "%(asctime)s %(levelname)s %(shortname)s - %(message)s",
    "file": None,
    "color": True,
}


def add_logging_args(parser) -> None:
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Logging level (e.g., INFO, DEBUG).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path.",
    )
    parser.add_argument(
        "--log-format",
        type=str,
        default=None,
        help="Console log format string.",
    )
    parser.add_argument(
        "--color",
        dest="log_color",
        action="store_true",
        help="Enable colored console logs.",
    )
    parser.add_argument(
        "--no-color",
        dest="log_color",
        action="store_false",
        help="Disable colored console logs.",
    )
    parser.set_defaults(log_color=None)


def _normalize_logging_config(
    config: Mapping[str, Any] | None
) -> dict[str, Any]:
    merged = dict(DEFAULT_LOGGING)
    if not config:
        return merged

    for key in ("level", "format", "file", "color"):
        if key in config and config[key] is not None:
            merged[key] = config[key]

    if "fmt_console" in config and config["fmt_console"] is not None:
        merged["format"] = config["fmt_console"]
    if "log_file" in config and config["log_file"] is not None:
        merged["file"] = config["log_file"]
    if "colored" in config and config["colored"] is not None:
        merged["color"] = config["colored"]

    return merged


def setup_logging_from_config(config: Mapping[str, Any] | None) -> None:
    log_cfg = _normalize_logging_config(config)
    setup_logging(
        log_cfg["level"],
        fmt_console=log_cfg["format"],
        log_file=log_cfg["file"],
        colored=log_cfg["color"],
    )

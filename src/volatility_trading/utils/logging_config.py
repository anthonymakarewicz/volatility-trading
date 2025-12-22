"""Project-wide logging configuration for scripts/entrypoints.

Design goals
- Library modules never call basicConfig; they just do `logger = getLogger(__name__)`.
- Scripts/application call `setup_logging(...)` once.
- Supports int and string levels (useful for CLI/env vars).
- Optional file logging.
- Optional per-module overrides.

Usage (in a script)

    from volatility_trading.utils.logging_config import setup_logging

    setup_logging(
        level="INFO",
        log_file="logs/orats_download.log",
        module_levels={
            "volatility_trading.data.orats_client_api": "WARNING",
        },
    )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping


class _ColorFormatter(logging.Formatter):
    """ANSI-colored formatter for console logs.

    Applies colors only to the *console* handler; file logs remain uncolored.
    """

    _RESET = "[0m"
    _LEVEL_COLOR: dict[int, str] = {
        logging.DEBUG: "[36m",        # cyan
        logging.INFO: "[32m",         # green
        logging.WARNING: "[33m",      # yellow
        logging.ERROR: "[31m",        # red
        logging.CRITICAL: "[1;31m",   # bold red
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self._LEVEL_COLOR.get(record.levelno)
        if not color:
            return super().format(record)

        original = record.levelname
        record.levelname = f"{color}{original}{self._RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


def _coerce_level(level: int | str) -> int:
    """Coerce a logging level given as int or string into an int."""
    if isinstance(level, int):
        return level

    s = str(level).strip().upper()
    if not s:
        raise ValueError("Empty logging level")

    if s.isdigit():
        return int(s)

    mapping: dict[str, int] = {
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }

    try:
        return mapping[s]
    except KeyError as e:
        raise ValueError(f"Unknown logging level: {level!r}") from e


# --------------
# Public API
# --------------

def setup_logging(
    level: int | str = "INFO",
    *,
    colored: bool = False,
    fmt: str = "%(asctime)s %(levelname)s %(name)s - %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    log_file: str | Path | None = None,
    module_levels: Mapping[str, int | str] | None = None,
    quiet_third_party: bool = True,
) -> None:
    """Configure logging (call once from scripts/entrypoints).

    Parameters
    - level: Root log level (int or string, e.g. logging.INFO or "INFO").
    - colored: If True, colorize console output (ANSI).
    - fmt/datefmt: Log format.
    - log_file: If provided, also write logs to this file.
    - module_levels: Optional per-logger overrides.
    - quiet_third_party: If True, reduce noise from common HTTP libs.

    Notes
    - Uses `force=True` so reruns (e.g. notebooks) don't duplicate handlers.
    """
    root_level = _coerce_level(level)
    handlers: list[logging.Handler] = []

    # Always add console handler
    console = logging.StreamHandler()
    if colored:
        console.setFormatter(_ColorFormatter(fmt=fmt, datefmt=datefmt))
    else:
        console.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    handlers.append(console)

    # Optionally add a file handler
    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        handlers.append(fh)

    # Global logging config
    logging.basicConfig(level=root_level, handlers=handlers, force=True)

    # Module specific logging config
    if module_levels:
        for name, lvl in module_levels.items():
            logging.getLogger(name).setLevel(_coerce_level(lvl))

    if quiet_third_party:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
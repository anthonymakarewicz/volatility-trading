"""Shared CLI helper utilities for app entrypoints."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def ensure_list(value: Any) -> list[Any] | None:
    """Normalize a scalar/iterable value into a list or `None`."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def add_print_config_arg(parser) -> None:
    """Add a `--print-config` flag to a parser."""
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print merged config (JSON) and exit.",
    )


def add_dry_run_arg(parser) -> None:
    """Add a `--dry-run` flag to a parser."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and log the plan without executing any actions.",
    )


def collect_logging_overrides(args) -> dict[str, Any]:
    """Collect logging override values from parsed CLI args."""
    overrides: dict[str, Any] = {}
    if getattr(args, "log_level", None):
        overrides["level"] = args.log_level
    if getattr(args, "log_file", None):
        overrides["file"] = args.log_file
    if getattr(args, "log_format", None):
        overrides["format"] = args.log_format
    if getattr(args, "log_color", None) is not None:
        overrides["color"] = args.log_color
    return overrides


def _normalize(obj: Any) -> Any:
    """Convert paths/mappings/sequences to JSON-serializable structures."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_normalize(v) for v in obj]
    return obj


def print_config(config: Mapping[str, Any]) -> None:
    """Pretty-print merged config as deterministic JSON."""
    normalized = _normalize(config)
    print(json.dumps(normalized, indent=2, sort_keys=True))


def log_dry_run(logger, plan: Mapping[str, Any]) -> None:
    """Log the dry-run plan as formatted JSON."""
    normalized = _normalize(plan)
    logger.info("DRY RUN: no actions were executed.")
    logger.info("DRY RUN plan:\n%s", json.dumps(normalized, indent=2, sort_keys=True))

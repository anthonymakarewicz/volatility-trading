from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def ensure_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def add_print_config_arg(parser) -> None:
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print merged config (JSON) and exit.",
    )


def collect_logging_overrides(args) -> dict[str, Any]:
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
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_normalize(v) for v in obj]
    return obj


def print_config(config: Mapping[str, Any]) -> None:
    normalized = _normalize(config)
    print(json.dumps(normalized, indent=2, sort_keys=True))

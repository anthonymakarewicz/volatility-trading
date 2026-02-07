from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import os

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    yaml = None


def add_config_arg(parser, *, default: str | None = None) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default=default,
        help="Path to a YAML config file.",
    )


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for --config. "
            "Install with `pip install pyyaml`."
        )

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(
            "Config file must contain a YAML mapping at the top level."
        )

    return data


def deep_merge(
    base: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    for key, value in base.items():
        if isinstance(value, Mapping):
            merged[key] = deep_merge(value, {})
        else:
            merged[key] = value

    for key, value in updates.items():
        if (
            isinstance(value, Mapping)
            and isinstance(merged.get(key), Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def build_config(
    defaults: Mapping[str, Any],
    yaml_path: str | Path | None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = deep_merge(defaults, load_yaml_config(yaml_path))
    if overrides:
        config = deep_merge(config, overrides)
    return config


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    return Path(expanded)

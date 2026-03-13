"""Config-loading and merging helpers for CLI entrypoints."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    yaml = None


def _repo_root() -> Path:
    """Return the repository root for repo-shipped config fallback lookup."""
    return Path(__file__).resolve().parents[3]


def resolve_repo_relative_path(value: str | Path | None) -> Path | None:
    """Resolve a path, with fallback to the repository root for relative paths.

    This keeps normal cwd-relative behavior, but allows repo-shipped config and
    data paths such as `config/...` and `data/...` to work from subdirectories.
    """
    p = resolve_path(value)
    if p is None:
        return None
    if not p.is_absolute() and not p.exists():
        repo_relative = _repo_root() / p
        if repo_relative.exists():
            return repo_relative
        if p.parts and p.parts[0] in {"config", "data", "reports"}:
            return repo_relative
    return p


def add_config_arg(parser, *, default: str | None = None) -> None:
    """Add a `--config` YAML-path argument to a parser."""
    parser.add_argument(
        "--config",
        type=str,
        default=default,
        help="Path to a YAML config file.",
    )


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    """Load a top-level YAML mapping config from disk."""
    if path is None:
        return {}

    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for --config. Install with `pip install pyyaml`."
        )

    p = resolve_repo_relative_path(path)
    if p is None:
        return {}

    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping at the top level.")

    return data


def deep_merge(
    base: Mapping[str, Any],
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge two mapping trees."""
    merged: dict[str, Any] = {}

    for key, value in base.items():
        if isinstance(value, Mapping):
            merged[key] = deep_merge(value, {})
        else:
            merged[key] = value

    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def build_config(
    defaults: Mapping[str, Any],
    yaml_path: str | Path | None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build final config via defaults <- YAML <- CLI overrides."""
    config = deep_merge(defaults, load_yaml_config(yaml_path))
    if overrides:
        config = deep_merge(config, overrides)
    return config


def resolve_path(value: str | Path | None) -> Path | None:
    """Resolve env/user-expanded path strings to `Path` objects."""
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    return Path(expanded)

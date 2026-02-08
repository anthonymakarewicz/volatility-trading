from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_load_yaml_config_none_returns_empty() -> None:
    mod = importlib.import_module("volatility_trading.cli.config")
    assert mod.load_yaml_config(None) == {}


def test_load_yaml_config_missing_file_raises(tmp_path: Path) -> None:
    mod = importlib.import_module("volatility_trading.cli.config")
    missing = tmp_path / "missing.yml"
    with pytest.raises(FileNotFoundError):
        mod.load_yaml_config(missing)


def test_load_yaml_config_non_mapping_raises(write_yaml) -> None:
    mod = importlib.import_module("volatility_trading.cli.config")
    path = write_yaml("bad.yml", ["a", "b"])
    with pytest.raises(ValueError, match="YAML mapping"):
        mod.load_yaml_config(path)


def test_load_yaml_config_reads_mapping(write_yaml) -> None:
    mod = importlib.import_module("volatility_trading.cli.config")
    path = write_yaml("ok.yml", {"a": 1, "b": {"c": 2}})
    assert mod.load_yaml_config(path) == {"a": 1, "b": {"c": 2}}


def test_load_yaml_config_requires_pyyaml(
    monkeypatch,
    write_yaml,
) -> None:
    mod = importlib.import_module("volatility_trading.cli.config")
    path = write_yaml("ok.yml", {"a": 1})
    monkeypatch.setattr(mod, "yaml", None)
    with pytest.raises(RuntimeError, match="PyYAML is required"):
        mod.load_yaml_config(path)


def test_deep_merge_merges_nested_and_overrides() -> None:
    mod = importlib.import_module("volatility_trading.cli.config")
    base = {"a": 1, "b": {"c": 1, "d": 2}, "e": [1, 2]}
    updates = {"b": {"c": 99}, "e": [3], "f": 5}
    merged = mod.deep_merge(base, updates)
    assert merged == {"a": 1, "b": {"c": 99, "d": 2}, "e": [3], "f": 5}


def test_build_config_precedence_defaults_yaml_overrides(write_yaml) -> None:
    mod = importlib.import_module("volatility_trading.cli.config")
    defaults = {"a": 1, "b": {"x": 1, "y": 2}, "lst": [1]}
    yaml_path = write_yaml("cfg.yml", {"a": 2, "b": {"y": 3}, "lst": [2]})
    overrides = {"b": {"x": 9}, "lst": [3], "c": 4}
    config = mod.build_config(defaults, yaml_path, overrides)
    assert config == {
        "a": 2,
        "b": {"x": 9, "y": 3},
        "lst": [3],
        "c": 4,
    }


def test_resolve_path_expands_home_and_env(monkeypatch, tmp_path: Path) -> None:
    mod = importlib.import_module("volatility_trading.cli.config")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))

    p1 = mod.resolve_path("~/raw")
    p2 = mod.resolve_path("$DATA_ROOT/inter")

    assert p1 == tmp_path / "raw"
    assert p2 == tmp_path / "data" / "inter"


def test_resolve_path_passthrough_and_none(tmp_path: Path) -> None:
    mod = importlib.import_module("volatility_trading.cli.config")
    assert mod.resolve_path(None) is None
    assert mod.resolve_path(tmp_path) == tmp_path

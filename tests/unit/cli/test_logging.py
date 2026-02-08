from __future__ import annotations

import importlib


def test_normalize_logging_config_defaults() -> None:
    mod = importlib.import_module("volatility_trading.cli.logging")
    normalized = mod._normalize_logging_config(None)
    assert normalized == mod.DEFAULT_LOGGING


def test_normalize_logging_config_overrides() -> None:
    mod = importlib.import_module("volatility_trading.cli.logging")
    cfg = {
        "level": "DEBUG",
        "format": "%(message)s",
        "file": "log.txt",
        "color": False,
    }
    normalized = mod._normalize_logging_config(cfg)
    assert normalized["level"] == "DEBUG"
    assert normalized["format"] == "%(message)s"
    assert normalized["file"] == "log.txt"
    assert normalized["color"] is False


def test_normalize_logging_config_legacy_keys_override() -> None:
    mod = importlib.import_module("volatility_trading.cli.logging")
    cfg = {
        "format": "new",
        "file": "new.log",
        "color": True,
        "fmt_console": "legacy",
        "log_file": "legacy.log",
        "colored": False,
    }
    normalized = mod._normalize_logging_config(cfg)
    assert normalized["format"] == "legacy"
    assert normalized["file"] == "legacy.log"
    assert normalized["color"] is False


def test_setup_logging_from_config_uses_normalized(monkeypatch) -> None:
    mod = importlib.import_module("volatility_trading.cli.logging")

    captured: dict[str, object] = {}

    def _setup_logging(level, *, fmt_console, log_file, colored):
        captured["level"] = level
        captured["fmt_console"] = fmt_console
        captured["log_file"] = log_file
        captured["colored"] = colored

    monkeypatch.setattr(mod, "setup_logging", _setup_logging)

    cfg = {
        "level": "WARNING",
        "fmt_console": "legacy",
        "log_file": "legacy.log",
        "colored": False,
    }
    mod.setup_logging_from_config(cfg)

    assert captured == {
        "level": "WARNING",
        "fmt_console": "legacy",
        "log_file": "legacy.log",
        "colored": False,
    }

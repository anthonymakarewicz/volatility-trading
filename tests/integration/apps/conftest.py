from __future__ import annotations

import json
from typing import Any

import pytest


@pytest.fixture
def parse_printed_config():
    def _parse(text: str) -> dict[str, Any]:
        return json.loads(text)

    return _parse


@pytest.fixture
def run_help(capsys):
    def _run(mod, expected: str) -> None:
        with pytest.raises(SystemExit) as exc:
            mod.main(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert expected in out

    return _run


@pytest.fixture
def run_print_config(capsys, parse_printed_config):
    def _run(mod, config_path: str) -> dict[str, Any]:
        mod.main(
            [
                "--config",
                config_path,
                "--print-config",
            ]
        )
        return parse_printed_config(capsys.readouterr().out)

    return _run


@pytest.fixture
def assert_paths_exist():
    def _assert(cfg: dict[str, Any], paths: list[tuple[str, ...]]) -> None:
        for keys in paths:
            cur: Any = cfg
            for key in keys:
                assert key in cur
                cur = cur[key]

    return _assert

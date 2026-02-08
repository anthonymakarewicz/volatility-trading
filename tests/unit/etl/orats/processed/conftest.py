from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass(frozen=True)
class DummyDF:
    """Small stand-in for the (df, out_path) return shape used by processed builders."""

    columns: list[str]
    height: int


@pytest.fixture
def dummy_df_factory():
    def _factory(*, columns: Sequence[str], height: int) -> DummyDF:
        return DummyDF(columns=list(columns), height=int(height))

    return _factory


@pytest.fixture
def manifest_writer(tmp_path: Path):
    """Capture manifest payloads written by builders without touching real data."""

    captured: dict[str, Any] = {}

    def _write_manifest_json(*, out_dir: Path, payload: dict[str, Any]) -> Path:
        captured["out_dir"] = out_dir
        captured["payload"] = payload

        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "manifest.json"
        path.write_text("{}", encoding="utf-8")
        return path

    return captured, _write_manifest_json

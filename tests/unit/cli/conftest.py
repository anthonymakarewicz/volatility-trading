from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def write_yaml(tmp_path: Path):
    def _write(name: str, data: Mapping[str, Any] | Any) -> Path:
        import yaml

        path = tmp_path / name
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)
        return path

    return _write

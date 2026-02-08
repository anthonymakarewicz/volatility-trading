from __future__ import annotations

import json
from typing import Any

import pytest


@pytest.fixture
def parse_printed_config():
    def _parse(text: str) -> dict[str, Any]:
        return json.loads(text)

    return _parse

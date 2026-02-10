"""Shared manifest writer for processed ORATS outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_manifest_json(*, out_dir: Path, payload: dict[str, Any]) -> Path:
    """Write a manifest.json sidecar next to the processed parquet.

    The manifest captures *how* the dataset was built (key parameters and
    switches) so downstream consumers (QC, backtests) can reliably
    reproduce/interpret the output.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)

    return path

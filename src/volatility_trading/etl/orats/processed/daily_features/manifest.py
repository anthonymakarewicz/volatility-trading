"""
Manifest helpers for processed daily-features panels.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_manifest_json(*, out_dir: Path, payload: dict) -> Path:
    """Write a manifest.json sidecar next to the processed parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)

    return path
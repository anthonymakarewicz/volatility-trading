"""
Manifest helpers for processed daily-features panels.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def build_manifest_payload(
    *,
    ticker: str,
    inter_root: Path,
    proc_root: Path,
    columns: list[str],
    n_rows_written: int,

    # dataset params
    endpoints: Sequence[str],
    endpoints_used: Sequence[str],
    missing_endpoints: Sequence[str],
    prefix_endpoint_cols: bool,
    priority_endpoints: Sequence[str] | None,

    # stats
    stats: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": "orats_daily_features",
        "ticker": str(ticker),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),

        "intermediate_root": str(inter_root),
        "processed_root": str(proc_root),

        "columns": list(columns),
        "n_rows_written": int(n_rows_written),

        "params": {
            "endpoints": list(endpoints),
            "endpoints_used": list(endpoints_used),
            "missing_endpoints": list(missing_endpoints),
            "prefix_endpoint_cols": bool(prefix_endpoint_cols),
            "priority_endpoints": (
                list(priority_endpoints) 
                if priority_endpoints is not None 
                else None
            ),
        },

        "stats": dict(stats) if stats is not None else {},
    }
"""volatility_trading.etl.orats.processed.options_chain.manifest

Manifest helpers for the ORATS options-chain builder.

This module is responsible for:
- building the manifest payload dict (build metadata)
- writing `manifest.json` next to the processed parquet output
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_manifest_payload(
    *,
    ticker: str,
    put_greeks_mode: str,
    exercise_style: str,
    merge_dividend_yield: bool,
    monies_implied_inter_root: Path | None,
    inter_root: Path,
    proc_root: Path,
    years: Any,
    dte_min: int,
    dte_max: int,
    moneyness_min: float,
    moneyness_max: float,
    columns: list[str],
    n_rows_written: int,
    stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": "orats_options_chain",
        "ticker": str(ticker),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "put_greeks_mode": str(put_greeks_mode),
        "exercise_style": str(exercise_style),
        "merge_dividend_yield": bool(merge_dividend_yield),
        "monies_implied_inter_root": (
            str(monies_implied_inter_root) if monies_implied_inter_root else None
        ),
        "inter_root": str(inter_root),
        "proc_root": str(proc_root),
        "years": [str(y) for y in years] if years is not None else None,
        "dte_min": int(dte_min),
        "dte_max": int(dte_max),
        "moneyness_min": float(moneyness_min),
        "moneyness_max": float(moneyness_max),
        "columns": list(columns),
        "n_rows_written": int(n_rows_written),
        "stats": dict(stats),
    }


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
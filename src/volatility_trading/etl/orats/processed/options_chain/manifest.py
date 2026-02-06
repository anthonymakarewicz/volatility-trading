"""volatility_trading.etl.orats.processed.options_chain.manifest

Manifest helpers for the ORATS options-chain builder.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_manifest_payload(
    *,
    ticker: str,
    inter_root: Path,
    proc_root: Path,
    columns: list[str],
    n_rows_written: int,

    # dataset params
    put_greeks_mode: str,
    exercise_style: str,
    merge_dividend_yield: bool,
    monies_implied_inter_root: Path | None,
    years: Any,
    dte_min: int,
    dte_max: int,
    moneyness_min: float,
    moneyness_max: float,

    # stats
    stats: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": "orats_options_chain",
        "ticker": str(ticker),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),

        # consistent roots naming
        "intermediate_root": str(inter_root),
        "processed_root": str(proc_root),

        # consistent output metadata
        "columns": list(columns),
        "n_rows_written": int(n_rows_written),

        # dataset-specific knobs live under params
        "params": {
            "put_greeks_mode": str(put_greeks_mode),
            "exercise_style": str(exercise_style),
            "merge_dividend_yield": bool(merge_dividend_yield),
            "monies_implied_inter_root": (
                str(monies_implied_inter_root)
                if monies_implied_inter_root
                else None
            ),
            "years": [str(y) for y in years] if years is not None else None,
            "dte_min": int(dte_min),
            "dte_max": int(dte_max),
            "moneyness_min": float(moneyness_min),
            "moneyness_max": float(moneyness_max),
        },

        # consistent stats behavior
        "stats": dict(stats) if stats is not None else {},
    }
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from .types import QCCheckResult, QCConfig, Severity

NA_STR = 'n/a'


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return NA_STR
    return f"{100.0 * x:.2f}%"


def _fmt_int(x: int | None) -> str:
    if x is None:
        return NA_STR
    return f"{x:,}"


def log_check(logger: logging.Logger, res: QCCheckResult) -> None:
    #  ---- INFO checks: metrics-only → log details instead of "viol=...". ----
    if res.severity == Severity.INFO:
        logger.info(
            "[%s|%s] %s rows=%s",
            res.severity.value,
            res.grade.value,
            res.name,
            _fmt_int(res.n_rows),
        )

        # Optional: print a compact metrics summary if available
        if res.details:
            # keep it stable and readable
            keys = sorted(res.details.keys())
            parts = []
            for k in keys:
                v = res.details[k]
                # format ints nicely
                if isinstance(v, int):
                    parts.append(f"{k}={v:,}")
                else:
                    parts.append(f"{k}={v}")
            logger.info("  metrics: %s", " | ".join(parts))

        return

    # ---- HARD/SOFT checks: standard violation summary ----
    logger.info(
        "[%s|%s] %s passed=%s viol=%s/%s (%s)",
        res.severity.value,
        res.grade.value,
        res.name,
        res.passed,
        _fmt_int(res.n_viol) if res.n_viol is not None else NA_STR,
        _fmt_int(res.n_rows) if res.n_rows is not None else NA_STR,
        _fmt_pct(res.viol_rate),
    )


def write_summary_json(path: Path, results: list[QCCheckResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [res.__dict__ for res in results]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def write_config_json(path: Path, config: QCConfig) -> None:
    """Write the QCConfig to a JSON sidecar file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
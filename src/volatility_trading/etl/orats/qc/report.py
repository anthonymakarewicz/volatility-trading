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


def _details_int(res: QCCheckResult, key: str) -> int | None:
    """Read an int-like field from details safely."""
    v = res.details.get(key) if res.details else None
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def log_check(logger: logging.Logger, res: QCCheckResult) -> None:
    # ---- INFO checks: metrics-only ----
    if res.severity == Severity.INFO:
        logger.info(
            "[%s|%s] %s rows=%s",
            res.severity.value,
            res.grade.value,
            res.name,
            _fmt_int(res.n_rows),
        )

        if res.details:
            keys = sorted(res.details.keys())
            parts: list[str] = []
            for k in keys:
                v = res.details[k]
                if isinstance(v, int):
                    parts.append(f"{k}={v:,}")
                else:
                    parts.append(f"{k}={v}")
            logger.info("  metrics: %s", " | ".join(parts))

        return

    # ---- HARD/SOFT checks: violation summary ----
    # Prefer dataset-level "units" if checker provided them.
    denom = res.n_units if res.n_units is not None else res.n_rows

    logger.info(
        "[%s|%s] %s passed=%s viol=%s/%s (%s)",
        res.severity.value,
        res.grade.value,
        res.name,
        res.passed,
        _fmt_int(res.n_viol),
        _fmt_int(denom),
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
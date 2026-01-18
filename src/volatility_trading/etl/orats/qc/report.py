from __future__ import annotations

import json
import logging
from pathlib import Path

from .types import QCCheckResult

NA_STR = 'n/a'


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return NA_STR
    return f"{100.0 * x:.2f}%"


def log_check(logger: logging.Logger, res: QCCheckResult) -> None:
    logger.info(
        "[%s|%s] %s passed=%s viol=%s/%s (%s)",
        res.severity.value,
        res.grade.value,
        res.name,
        res.passed,
        res.n_viol if res.n_viol is not None else NA_STR,
        res.n_rows if res.n_rows is not None else NA_STR,
        _fmt_pct(res.viol_rate),
    )


def write_summary_json(path: Path, results: list[QCCheckResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [res.__dict__ for res in results]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .endpoints import get_endpoint_spec

logger = logging.getLogger(__name__)


def jitter_sleep(base_s: float) -> float:
    """Apply random jitter to backoff sleeps."""
    return base_s * (0.7 + 0.6 * random.random())


def is_missing_param_value(v: Any) -> bool:
    """Return True if a param value should be considered missing/empty."""
    if v is None:
        return True
    if isinstance(v, str):
        return len(v.strip()) == 0
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
        return len([x for x in v if x is not None and str(x).strip()]) == 0
    return False


def orats_list_param(values: Iterable[str] | None) -> str | None:
    """Normalize list-like ORATS params into a comma-separated string."""
    if values is None:
        return None

    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s in seen:
            continue
        out.append(s)
        seen.add(s)

    return ",".join(out) if out else None


def normalize_orats_params(params: Mapping[str, Any]) -> dict[str, str]:
    """Convert higher-level params into ORATS wire-format query parameters."""
    out: dict[str, str] = {}

    for k, v in params.items():
        if v is None:
            continue

        if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
            joined = orats_list_param(v)  # type: ignore[arg-type]
            if joined is not None:
                out[str(k)] = joined
            continue

        out[str(k)] = str(v)

    return out


def params_summary(wire_params: Mapping[str, str]) -> str:
    """Create a safe, compact params summary for logs (never includes token)."""
    keys = [k for k in wire_params.keys() if k != "token"]
    keys.sort()
    parts: list[str] = []

    for k in keys:
        if k in {"ticker", "tradeDate", "expirDate"}:
            parts.append(f"{k}={wire_params.get(k)}")
        elif k == "fields":
            v = wire_params.get(k, "")
            n = 0 if not v else len([x for x in v.split(",") if x.strip()])
            parts.append(f"fields={n}")
        else:
            parts.append(k)

    return ",".join(parts)


def validate_endpoint_params(endpoint: str, params: Mapping[str, Any]) -> None:
    """Ensure required params for an endpoint are present and non-empty."""
    spec = get_endpoint_spec(endpoint)
    missing: list[str] = []
    for k in spec.required:
        if k not in params or is_missing_param_value(params.get(k)):
            missing.append(k)

    if missing:
        raise ValueError(
            f"Missing required params for endpoint '{endpoint}': {missing}. "
            f"Required: {spec.required}"
        )

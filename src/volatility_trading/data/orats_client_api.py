from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl
import requests

from .orats_api_endpoints import get_endpoint_spec

logger = logging.getLogger(__name__)

ORATS_BASE_URL = "https://api.orats.io"


# ----------------------------
# Small utilities
# ----------------------------

def _orats_list_param(values: Iterable[str] | None) -> str | None:
    """Convert a list/iterable of ORATS values into a comma-separated string.

    Use for request params like:
      - ticker="SPX,NDX,VIX"
      - fields="tradeDate,expirDate,calVol"
    """
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


def _validate_endpoint_params(endpoint: str, params: Mapping[str, Any]) -> None:
    """Validate required params for a given endpoint before sending a request."""
    spec = get_endpoint_spec(endpoint)
    missing: list[str] = []
    for k in spec.required:
        if k not in params or _is_missing_param_value(params.get(k)):
            missing.append(k)
    if missing:
        raise ValueError(
            f"Missing required params for endpoint '{endpoint}': {missing}. "
            f"Required: {spec.required}"
        )
    

def _normalize_orats_params(params: Mapping[str, Any]) -> dict[str, str]:
    """Normalize ORATS query params.

    Allows higher-level code to pass params naturally, e.g.
        {
            "ticker": ["SPX", "NDX"],
            "tradeDate": "2019-11-29",
            "fields": ["tradeDate", "expirDate"],
        }

    And outputs the expected ORATS API wire format:
        {
            "ticker": "SPX,NDX",
            "tradeDate": "2019-11-29",
            "fields": "tradeDate,expirDate",
        }
    """
    out: dict[str, str] = {}

    for k, v in params.items():
        if v is None:
            continue

        # list/tuple of strings -> comma-separated
        if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
            joined = _orats_list_param(v)  # type: ignore[arg-type]
            if joined is not None:
                out[str(k)] = joined
            continue

        out[str(k)] = str(v)

    return out


def _params_summary(wire_params: Mapping[str, str]) -> str:
    """Small, safe summary for logs/errors (never includes token)."""
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


def _is_missing_param_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return len(v.strip()) == 0
    # lists/tuples of values (e.g., tickers/fields)
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
        return len([x for x in v if x is not None and str(x).strip()]) == 0
    return False


def _orats_payload_to_polars(payload: dict[str, Any]) -> pl.DataFrame:
    if "data" not in payload:
        raise ValueError(f"Unexpected ORATS payload keys: {list(payload.keys())}")

    data = payload["data"]
    if not data:
        return pl.DataFrame()

    cleaned = [{k.strip(): v for k, v in row.items()} for row in data]
    return pl.DataFrame(cleaned)


# ----------------------------
# HTTP client
# ----------------------------

@dataclass(frozen=True)
class OratsClient:
    """HTTP GET client for ORATS API with retries and safe logging."""

    token: str
    base_url: str = ORATS_BASE_URL
    timeout_s: float = 30.0
    max_retries: int = 5
    backoff_s: float = 0.75

    def _get(
        self,
        path: str,
        params: Mapping[str, Any],
        *,
        session: requests.Session | None = None,
    ) -> requests.Response:
        """Low-level GET with retries.

        Retries on: 429, 5xx, timeouts / transient connection errors.
        Does NOT retry on: other 4xx (bad params/auth/etc.).
        Closes an internally-created Session.
        """
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")

        created_session = session is None
        sess = session or requests.Session()

        wire_params = _normalize_orats_params(params)
        params_log = _params_summary(wire_params)
        wire_params["token"] = self.token

        last_err: Exception | None = None

        try:
            for attempt in range(self.max_retries + 1):
                try:
                    logger.debug(
                        "ORATS GET attempt=%d/%d path=%s params=[%s]",
                        attempt + 1,
                        self.max_retries + 1,
                        path,
                        params_log,
                    )

                    resp = sess.get(
                        url, 
                        params=wire_params, 
                        timeout=self.timeout_s
                    )

                    # Fail fast on permanent client errors (except 429)
                    if 400 <= resp.status_code <= 499 and resp.status_code != 429:
                        resp.raise_for_status()

                    # Retry on rate limit / server errors
                    if resp.status_code == 429 or 500 <= resp.status_code <= 599:
                        ra = resp.headers.get("Retry-After")
                        if ra is not None:
                            try:
                                sleep_s = float(ra)
                            except ValueError:
                                sleep_s = self.backoff_s * (2**attempt)
                        else:
                            sleep_s = self.backoff_s * (2**attempt)

                        sleep_s = min(30.0, sleep_s)

                        if attempt >= self.max_retries:
                            resp.raise_for_status()

                        logger.warning(
                            "ORATS retryable status=%s path=%s params=[%s] "
                            "sleep_s=%.2f",
                            resp.status_code,
                            path,
                            params_log,
                            sleep_s,
                        )
                        time.sleep(sleep_s)
                        continue

                    resp.raise_for_status()

                    logger.debug(
                        "ORATS GET success status=%s path=%s params=[%s]",
                        resp.status_code,
                        path,
                        params_log,
                    )
                    return resp

                except requests.exceptions.HTTPError as e:
                    last_err = e
                    r = getattr(e, "response", None)
                    if r is not None:
                        sc = r.status_code
                    else:
                        sc = None

                    # Non-429 4xx: do not retry
                    if sc is not None and 400 <= sc <= 499 and sc != 429:
                        logger.error(
                            "ORATS HTTPError status=%s path=%s params=[%s]",
                            sc,
                            path,
                            params_log,
                        )
                        break

                    if attempt >= self.max_retries:
                        break

                    sleep_s = min(30.0, self.backoff_s * (2**attempt))
                    logger.warning(
                        "ORATS HTTPError retrying path=%s params=[%s] "
                        "sleep_s=%.2f",
                        path,
                        params_log,
                        sleep_s,
                    )
                    time.sleep(sleep_s)

                except (
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                ) as e:
                    last_err = e
                    if attempt >= self.max_retries:
                        break

                    sleep_s = min(30.0, self.backoff_s * (2**attempt))
                    logger.warning(
                        "ORATS transport error retrying path=%s params=[%s] "
                        "sleep_s=%.2f err=%r",
                        path,
                        params_log,
                        sleep_s,
                        e,
                    )
                    time.sleep(sleep_s)

                except Exception as e:
                    last_err = e
                    if attempt >= self.max_retries:
                        break

                    sleep_s = min(30.0, self.backoff_s * (2**attempt))
                    logger.warning(
                        "ORATS unexpected error retrying path=%s params=[%s] "
                        "sleep_s=%.2f err=%r",
                        path,
                        params_log,
                        sleep_s,
                        e,
                    )
                    time.sleep(sleep_s)
        finally:
            if created_session:
                sess.close()

        logger.error(
            "ORATS GET failed after retries path=%s params=[%s]",
            path,
            params_log,
        )
        raise RuntimeError(
            f"ORATS GET failed after retries: path={path} params=[{params_log}]"
        ) from last_err

    def get_df(
        self,
        endpoint: str,
        params: Mapping[str, Any],
        *,
        session: requests.Session | None = None,
    ) -> pl.DataFrame:
        """GET endpoint and return Polars DataFrame from its JSON payload."""
        _validate_endpoint_params(endpoint, params)
        spec = get_endpoint_spec(endpoint)
        resp = self._get(spec.path, params, session=session)
        payload = resp.json()
        return _orats_payload_to_polars(payload)
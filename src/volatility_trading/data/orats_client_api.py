"""ORATS API client utilities.

This module provides a small, typed HTTP client (`OratsClient`) plus helpers for
validating and normalizing request parameters. It is designed to be used by
downloader/orchestrator code that snapshots ORATS API responses to disk.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import requests

from .orats_api_endpoints import get_endpoint_spec

logger = logging.getLogger(__name__)

ORATS_BASE_URL = "https://api.orats.io"


# ----------------------------------------------------------------------------
# Private Helpers
# ----------------------------------------------------------------------------

def _orats_list_param(values: Iterable[str] | None) -> str | None:
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


def _validate_endpoint_params(endpoint: str, params: Mapping[str, Any]) -> None:
    """Ensure required params for an endpoint are present and non-empty."""
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
    """Convert higher-level params into ORATS wire-format query parameters."""
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
    """Create a safe, compact params summary for logs (never includes the token)."""
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
    """Return True if a param value should be considered missing/empty."""
    if v is None:
        return True
    if isinstance(v, str):
        return len(v.strip()) == 0
    # lists/tuples of values (e.g., tickers/fields)
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
        return len([x for x in v if x is not None and str(x).strip()]) == 0
    return False


# ----------------------------------------------------------------------------
# Public API (HTTP client)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class OratsClient:
    """HTTP client for ORATS API requests.

    The client wraps `requests` and provides:
    - endpoint-aware parameter validation (required args per endpoint)
    - normalization of list-like params into ORATS' comma-separated wire format
    - retries with exponential backoff for transient failures

    Retry policy (high level):
    - Retries on HTTP 429 (rate limit) and HTTP 5xx (server errors)
    - Retries on transport-level issues (timeouts / connection errors)
    - Does *not* retry on other HTTP 4xx responses (bad params, auth, etc.)

    Session handling:
    - You may pass a shared `requests.Session` to reuse connections across many calls.
    - If you do not pass a session, the client creates one internally and closes it.

    Logging:
    - Request logs use a redacted params summary that never includes the API token.

    Exceptions:
    - Non-429 HTTP 4xx errors raise `requests.HTTPError` (fail-fast).
    - If all retries are exhausted for a retryable failure, the client raises
      `RuntimeError` with the original exception attached as `__cause__`.
    """
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
        """Low-level GET with retries for transient failures (429/5xx/transport)."""
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
                    if (
                        400 <= resp.status_code <= 499
                        and resp.status_code != 429
                    ):
                        resp.raise_for_status()

                    # Retry on rate limit / server errors
                    if (
                        resp.status_code == 429
                        or 500 <= resp.status_code <= 599
                    ):
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

    def get_payload(
        self,
        endpoint: str,
        params: Mapping[str, Any],
        *,
        session: requests.Session | None = None,
    ) -> dict[str, Any]:
        """Fetch an ORATS endpoint and return the raw JSON payload.

        Parameters
        ----------
        endpoint:
            Logical endpoint name (key in `orats_api_endpoints.ENDPOINTS`).
            The client resolves this to an HTTP path and 
            validates required parameters.
        params:
            Query parameters for the endpoint. Values may be passed naturally
            (e.g. lists for `ticker` / `fields`), and will be normalized 
            into ORATS' expected comma-separated wire format.
        session:
            Optional shared `requests.Session` for connection reuse.

        Returns
        -------
        dict
            The parsed JSON payload returned by ORATS (typically containing 
            a `data` list plus metadata fields).

        Raises
        ------
        ValueError
            If required endpoint parameters are missing/empty.
        requests.HTTPError
            For non-429 HTTP 4xx responses (fail-fast).
        RuntimeError
            If retries are exhausted for a retryable failure (429/5xx/transport).
        json.JSONDecodeError
            If the response body is not valid JSON.
        TypeError
            If the decoded JSON is not a JSON object (dict).
        """
        _validate_endpoint_params(endpoint, params)
        spec = get_endpoint_spec(endpoint)

        resp = self._get(spec.path, params, session=session)

        # Let json() raise if the payload is not valid JSON
        payload = resp.json()
        if not isinstance(payload, dict):
            raise TypeError(
                f"Expected ORATS JSON payload to be an object/dict, got {type(payload)}"
            )
        return payload
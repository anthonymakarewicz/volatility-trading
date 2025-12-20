from __future__ import annotations

import requests
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl


# ----------------------------
# Configuration
# ----------------------------

ORATS_BASE_URL = "https://api.orats.io"

@dataclass(frozen=True)
class EndpointSpec:
    path: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()


ENDPOINTS: dict[str, EndpointSpec] = {
    "monies_implied": EndpointSpec(
        path="/datav2/hist/monies/implied",
        required=("ticker", "tradeDate"),
        optional=("fields",),
    ),
    "cores": EndpointSpec(
        path="/datav2/hist/cores",
        required=("ticker", "tradeDate"),
        optional=("fields",),
    ),
    "summaries": EndpointSpec(
        path="/datav2/hist/summaries",
        required=("ticker", "tradeDate"),
        optional=("fields",),
    ),
}


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


def _normalize_orats_params(params: Mapping[str, Any]) -> dict[str, str]:
    """Normalize ORATS query params.
    This lets higher-level code pass params naturally, e.g.
        {"ticker": ["SPX", "NDX"], "tradeDate": "2019-11-29", "fields": ["tradeDate", "expirDate"]}

    And output the expected format by ORATS API:
        {"ticker": "SPX,NDX", "tradeDate": "2019-11-29", "fields": "tradeDate,expirDate"}
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


def _orats_payload_to_polars(payload: dict) -> pl.DataFrame:
    data = payload.get("data", [])
    if not data:
        return pl.DataFrame()

    cleaned = [{k.strip(): v for k, v in row.items()} for row in data]
    df = pl.DataFrame(cleaned)

    return df


def _get_endpoint_spec(endpoint: str) -> EndpointSpec:
    """Return the spec (path + required params) for a supported ORATS endpoint name."""
    try:
        return ENDPOINTS[endpoint]
    except KeyError as e:
        supported = ", ".join(sorted(ENDPOINTS.keys()))
        raise KeyError(f"Unknown ORATS endpoint '{endpoint}'. Supported: {supported}") from e


def _is_missing_param_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return len(v.strip()) == 0
    # lists/tuples of values (e.g., tickers/fields)
    if isinstance(v, Sequence) and not isinstance(v, (str, bytes, bytearray)):
        return len([x for x in v if x is not None and str(x).strip()]) == 0
    return False


def _validate_endpoint_params(endpoint: str, params: Mapping[str, Any]) -> None:
    """Validate required params for a given endpoint before sending an HTTP request."""
    spec = _get_endpoint_spec(endpoint)
    missing: list[str] = []
    for k in spec.required:
        if k not in params or _is_missing_param_value(params.get(k)):
            missing.append(k)
    if missing:
        raise ValueError(
            f"Missing required params for endpoint '{endpoint}': {missing}. "
            f"Required: {spec.required}"
        )


# ----------------------------
# HTTP client
# ----------------------------

@dataclass(frozen=True)
class OratsClient:
    token: str
    base_url: str = ORATS_BASE_URL  
    timeout_s: float = 30.0
    max_retries: int = 5
    backoff_s: float = 0.75  # exponential-ish with jitter

    def _get(
        self,
        path: str,
        params: Mapping[str, Any],
        *,
        session: requests.Session | None = None,
    ) -> requests.Response:
        """
        Low-level GET with retries. Returns requests.Response on success.
        Retries on: 429, 5xx, timeouts / transient connection errors.
        Does NOT retry on: other 4xx (bad params/auth/etc.).
        Closes an internally-created Session.
        """
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")

        created_session = session is None
        sess = session or requests.Session()

        # normalize params (lists -> comma-separated strings) and always include token
        wire_params = _normalize_orats_params(params)
        wire_params["token"] = self.token

        last_err: Exception | None = None

        try:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = sess.get(url, params=wire_params, timeout=self.timeout_s)

                    # Permanent client errors (except 429) -> fail fast (no retry)
                    if 400 <= resp.status_code <= 499 and resp.status_code != 429:
                        resp.raise_for_status()

                    # Retry on rate-limit / server errors
                    if resp.status_code == 429 or 500 <= resp.status_code <= 599:
                        ra = resp.headers.get("Retry-After")
                        if ra is not None:
                            try:
                                sleep_s = float(ra)
                            except ValueError:
                                sleep_s = self.backoff_s * (2 ** attempt)
                        else:
                            sleep_s = self.backoff_s * (2 ** attempt)

                        time.sleep(min(30.0, sleep_s))
                        continue

                    resp.raise_for_status()
                    return resp

                except requests.exceptions.HTTPError as e:
                    last_err = e
                    # If it's a non-429 4xx, do not retry
                    r = getattr(e, "response", None)
                    if r is not None and 400 <= r.status_code <= 499 and r.status_code != 429:
                        break

                    if attempt >= self.max_retries:
                        break
                    time.sleep(min(30.0, self.backoff_s * (2 ** attempt)))

                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    last_err = e
                    if attempt >= self.max_retries:
                        break
                    time.sleep(min(30.0, self.backoff_s * (2 ** attempt)))

                except Exception as e:
                    last_err = e
                    if attempt >= self.max_retries:
                        break
                    time.sleep(min(30.0, self.backoff_s * (2 ** attempt)))
        finally:
            if created_session:
                sess.close()

        raise RuntimeError(f"ORATS GET failed after retries: {url} params={wire_params}") from last_err
    
    def get_df(
        self,
        endpoint: str,
        params: Mapping[str, Any],
        *,
        session: requests.Session | None = None,
    ) -> pl.DataFrame:
        """GET endpoint and return Polars DataFrame from its JSON payload."""
        _validate_endpoint_params(endpoint, params)
        spec = _get_endpoint_spec(endpoint)
        resp = self._get(spec.path, params, session=session)
        payload = resp.json()
        df = _orats_payload_to_polars(payload)
        return df
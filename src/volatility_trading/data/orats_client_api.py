from __future__ import annotations

import json
import requests
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import polars as pl


# ----------------------------
# Small utilities
# ----------------------------

def orats_fields_param(fields: Iterable[str] | None) -> str | None:
    """
    Convert a list/iterable of ORATS tickers/fields into the comma-separated `fields=` string.
    - Strips whitespace
    - Drops empties / None
    - Dedupes while preserving order
    """
    if fields is None:
        return None

    out: list[str] = []
    seen: set[str] = set()
    for f in fields:
        if f is None:
            continue
        f = str(f).strip()
        if not f or f in seen:
            continue
        out.append(f)
        seen.add(f)

    return ",".join(out) if out else None


def orats_payload_to_polars(payload: dict) -> pl.DataFrame:
    data = payload.get("data", [])
    if not data:
        return pl.DataFrame()

    cleaned = [{k.strip(): v for k, v in row.items()} for row in data]
    df = pl.DataFrame(cleaned)

    return df


# ----------------------------
# HTTP client
# ----------------------------

@dataclass(frozen=True)
class OratsClient:
    token: str
    base_url: str = "https://api.orats.com"   # adjust if your endpoint differs
    timeout_s: float = 30.0
    max_retries: int = 5
    backoff_s: float = 0.75  # exponential-ish with jitter

    def _get(
        self,
        path: str,
        params: dict[str, str],
        *,
        session: requests.Session | None = None,
    ) -> requests.Response:
        """
        Low-level GET with retries. Raises on final failure.
        """
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
        sess = session or requests.Session()

        # always include token
        params = dict(params)
        params["token"] = self.token

        last_err: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = sess.get(url, params=params, timeout=self.timeout_s)

                # Retry on 429 / 5xx
                if resp.status_code == 429 or 500 <= resp.status_code <= 599:
                    # try respect Retry-After if present
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

            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                time.sleep(min(30.0, self.backoff_s * (2 ** attempt)))

        raise RuntimeError(f"ORATS GET failed after retries: {url} params={params}") from last_err

    def get_df(
        self,
        path: str,
        params: dict[str, Any],
        *,
        session: requests.Session | None = None,
    ) -> pl.DataFrame:
        """GET endpoint and return Polars DataFrame from its JSON payload."""
        resp = self._get(path, params, session=session)
        return orats_payload_to_polars(resp.json())
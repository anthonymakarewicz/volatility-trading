from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import requests

from ..client import OratsClient
from ..endpoints import DownloadStrategy, get_endpoint_spec
from ..io import (
    ALLOWED_COMPRESSIONS,
    DEFAULT_COMPRESSION,
    validate_years,
)

from ._handlers import DOWNLOAD_HANDLERS
from ._helpers import unique_preserve_order
from ..types import DownloadApiResult

logger = logging.getLogger(__name__)


def download(
    *,
    token: str,
    endpoint: str,
    raw_root: str | Path,
    tickers: Iterable[str],
    year_whitelist: Iterable[int] | Iterable[str] | None = None,
    fields: Sequence[str] | None = None,
    compression: str = DEFAULT_COMPRESSION,
    sleep_s: float = 0.0,
    overwrite: bool = False,
) -> DownloadApiResult:
    """
    Download an ORATS API endpoint and store raw JSON payload snapshots.

    This is an ingestion utility that writes *one JSON file per API request*,
    preserving the response as returned by ORATS ("raw" layer). Downstream
    pipelines can later parse these snapshots into curated parquet datasets.

    Strategy
    --------
    The endpoint determines the download strategy via `get_endpoint_spec(endpoint)`:

    - BY_TRADE_DATE:
        Iterates over NYSE trading sessions (XNYS) for the requested years, and
        downloads one payload per (tradeDate, ticker_chunk). Tickers are chunked
        to respect ORATS list-parameter size limits.

    - FULL_HISTORY:
        Downloads one payload per ticker (no `tradeDate`). Any `year_whitelist`
        passed by the caller is ignored (and a warning is logged).

    Storage layout
    --------------
    - BY_TRADE_DATE:
        raw_root/endpoint=<endpoint>/year=YYYY/YYYY-MM-DD_chunk000.json.gz

    - FULL_HISTORY:
        raw_root/endpoint=<endpoint>/underlying=<TICKER>/data.json.gz

    Parameters
    ----------
    token:
        ORATS API token.
    endpoint:
        Logical endpoint name (key in `orats_api_endpoints.ENDPOINTS`).
    raw_root:
        Root folder where raw snapshots will be written.
    tickers:
        Sequence of tickers. Values are stripped, de-duplicated (preserving first
        occurrence), and then (for BY_TRADE_DATE) chunked into <= MAX_PER_CALL.
    year_whitelist:
        Years to download for BY_TRADE_DATE endpoints. Required for BY_TRADE_DATE
        and ignored for FULL_HISTORY.
    fields:
        Optional list of ORATS fields to request. If None, ORATS returns its
        default field set for the endpoint.
    compression:
        Compression mode for raw JSON snapshots: "gz" (gzip) or "none".
    sleep_s:
        Optional polite delay (seconds) between requests.
    overwrite:
        If False (default), existing output files are skipped. If True, existing
        files are replaced.

    Notes
    -----
    - JSON payloads are written even when `payload['data']` is empty.
    - No `.empty`/`.error` markers are written.
    - Transient failures are logged and recorded in the returned result; the
      downloader continues.
    - Fatal failures (e.g., non-429 4xx, bad params, non-JSON responses) are
      re-raised to stop the run.
    """
    raw_root_p = Path(raw_root)

    if compression not in ALLOWED_COMPRESSIONS:
        raise ValueError(
            f"Unsupported compression '{compression}'. Allowed: "
            f"{sorted(ALLOWED_COMPRESSIONS)}"
        )

    tickers_clean = [
        str(t).strip()
        for t in tickers
        if t is not None and str(t).strip()
    ]
    tickers_clean = unique_preserve_order(tickers_clean)
    if not tickers_clean:
        raise ValueError("tickers must be non-empty")

    years_list: list[int] | None
    if year_whitelist is None:
        years_list = None
    else:
        years_list = [int(y) for y in year_whitelist]

    spec = get_endpoint_spec(endpoint)
    client = OratsClient(token=token)

    handler = DOWNLOAD_HANDLERS.get(spec.strategy, None)
    if handler is None:
        raise ValueError(
            f"No download handler registered for strategy: {spec.strategy}"
        )

    with requests.Session() as session:
        if spec.strategy == DownloadStrategy.FULL_HISTORY:
            if year_whitelist is not None:
                logger.warning(
                    "year_whitelist is ignored for endpoint=%s (FULL_HISTORY)",
                    endpoint,
                )
            return handler(
                client=client,
                session=session,
                endpoint=endpoint,
                raw_root=raw_root_p,
                tickers=tickers_clean,
                fields=fields,
                sleep_s=sleep_s,
                overwrite=overwrite,
                compression=compression,
            )

        if years_list is None:
            raise ValueError(
                "year_whitelist must be provided for BY_TRADE_DATE endpoints"
            )
        years = validate_years(years_list)

        return handler(
            client=client,
            session=session,
            endpoint=endpoint,
            raw_root=raw_root_p,
            tickers=tickers_clean,
            years=years,
            fields=fields,
            sleep_s=sleep_s,
            overwrite=overwrite,
            compression=compression,
        )
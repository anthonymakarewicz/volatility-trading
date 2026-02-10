"""Download step for ORATS HTTP API raw payload snapshots.

Reads ORATS endpoint payloads and writes one raw JSON snapshot per request.
"""

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
from ..types import DownloadApiResult
from ._handlers import DOWNLOAD_HANDLERS
from ._helpers import unique_preserve_order

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
    """Download one ORATS endpoint and store raw JSON payload snapshots.

    Args:
        token: ORATS API token.
        endpoint: Logical endpoint name from the endpoint registry.
        raw_root: Root directory where raw snapshots are written.
        tickers: Requested ticker universe; cleaned and de-duplicated in-order.
        year_whitelist: Required for BY_TRADE_DATE endpoints; ignored otherwise.
        fields: Optional subset of endpoint fields.
        compression: Raw JSON compression mode (`"gz"` or `"none"`).
        sleep_s: Delay in seconds between requests.
        overwrite: Replace existing raw snapshots when `True`.

    Returns:
        Summary including request counts, written paths, failures, and duration.

    Raises:
        ValueError: For invalid compression, empty tickers, or missing years.
    """
    raw_root_p = Path(raw_root)

    if compression not in ALLOWED_COMPRESSIONS:
        raise ValueError(
            f"Unsupported compression '{compression}'. Allowed: "
            f"{sorted(ALLOWED_COMPRESSIONS)}"
        )

    tickers_clean = [
        str(t).strip() for t in tickers if t is not None and str(t).strip()
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

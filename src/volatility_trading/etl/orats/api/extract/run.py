from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from ..endpoints import DownloadStrategy, get_endpoint_spec
from ..io import ALLOWED_COMPRESSIONS, validate_years

from ._handlers import extract_by_trade_date, extract_full_history
from ..types import ExtractApiResult

logger = logging.getLogger(__name__)


def extract(
    *,
    endpoint: str,
    raw_root: str | Path,
    intermediate_root: str | Path,
    tickers: Iterable[str] | None = None,
    year_whitelist: Iterable[int] | Iterable[str] | None = None,
    compression: str = "gz",
    overwrite: bool = False,
    parquet_compression: str = "zstd",
) -> ExtractApiResult:
    """
    Extract ORATS raw API snapshots into intermediate parquet.

    Parameters
    ----------
    endpoint:
        Endpoint name (key in `orats_api_endpoints.ENDPOINTS`).
    raw_root:
        Root of raw snapshots produced by `orats_downloader_api.download`.
    intermediate_root:
        Root of intermediate parquet output.
    tickers:
        Optional ticker allowlist. If None:
          - FULL_HISTORY: inferred from `underlying=*` folders.
          - BY_TRADE_DATE: keep all tickers present in payloads.
    year_whitelist:
        Required for BY_TRADE_DATE endpoints. Ignored for FULL_HISTORY.
    compression:
        "gz" or "none" (must match how raw snapshots were written).
    overwrite:
        If False (default), skip intermediate files that already exist.
    parquet_compression:
        Parquet compression (e.g. "zstd", "snappy").

    Returns
    -------
    ExtractApiResult:
        Summary including written files and failures.
    """
    raw_root_p = Path(raw_root)
    interm_root_p = Path(intermediate_root)

    if compression not in ALLOWED_COMPRESSIONS:
        raise ValueError(
            f"Unsupported compression '{compression}'. Allowed: "
            f"{sorted(ALLOWED_COMPRESSIONS)}"
        )

    tickers_clean: list[str] | None
    if tickers is None:
        tickers_clean = None
    else:
        tickers_clean = [
            str(t).strip()
            for t in tickers
            if t is not None and str(t).strip()
        ]
        # De-duplicate while preserving order
        seen: set[str] = set()
        tickers_clean = [t for t in tickers_clean if not (t in seen or seen.add(t))]
        if not tickers_clean:
            raise ValueError("tickers is passed but none of them is valid")

    spec = get_endpoint_spec(endpoint)

    if spec.strategy == DownloadStrategy.FULL_HISTORY:
        if year_whitelist is not None:
            logger.warning(
                "year_whitelist is ignored for endpoint=%s (FULL_HISTORY)",
                endpoint,
            )
        return extract_full_history(
            endpoint=endpoint,
            raw_root=raw_root_p,
            intermediate_root=interm_root_p,
            tickers=tickers_clean,
            compression=compression,
            overwrite=overwrite,
            parquet_compression=parquet_compression,
        )

    if year_whitelist is None:
        raise ValueError(
            "year_whitelist must be provided for BY_TRADE_DATE endpoints"
        )

    years = validate_years(year_whitelist)

    return extract_by_trade_date(
        endpoint=endpoint,
        raw_root=raw_root_p,
        intermediate_root=interm_root_p,
        tickers=tickers_clean,
        years=years,
        compression=compression,
        overwrite=overwrite,
        parquet_compression=parquet_compression,
    )

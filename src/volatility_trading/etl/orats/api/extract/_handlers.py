from __future__ import annotations

import logging
import time
from pathlib import Path

import polars as pl

from ..endpoints import DownloadStrategy
from ..io import (
    intermediate_full_history,
    raw_by_trade_date_dir,
    raw_full_history_dir,
    json_suffix,
)

from ._helpers import (
    apply_endpoint_schema,
    glob_raw_suffix,
    is_non_fatal_extract_error,
    load_payload,
    payload_to_df,
    remove_duplicates,
    write_parquet_atomic,
)
from ..types import ExtractApiResult

logger = logging.getLogger(__name__)


def extract_full_history(
    *,
    endpoint: str,
    raw_root: Path,
    intermediate_root: Path,
    tickers: list[str] | None,
    compression: str,
    overwrite: bool,
    parquet_compression: str,
) -> ExtractApiResult:
    """Extract FULL_HISTORY raw payloads (one file per ticker)."""
    t0 = time.perf_counter()
    out_paths: list[Path] = []
    failed: list[Path] = []
    n_seen = 0
    n_read = 0
    n_rows = 0

    base = raw_root / f"endpoint={endpoint}"
    if not base.exists():
        raise FileNotFoundError(f"Raw endpoint folder not found: {base}")

    if tickers is None:
        tickers = sorted(
            p.name.split("=", 1)[1]
            for p in base.glob("underlying=*")
            if p.is_dir()
        )

    logger.info(
        "Extract FULL_HISTORY endpoint=%s tickers=%d",
        endpoint,
        len(tickers),
    )

    for ticker in tickers:
        raw_dir = raw_full_history_dir(raw_root, endpoint, ticker)
        fp = raw_dir / f"data{json_suffix(compression)}"

        if not fp.exists():
            logger.debug(
                "Missing raw snapshot endpoint=%s ticker=%s expected=%s",
                endpoint,
                ticker,
                fp,
            )
            continue

        other = raw_dir / ("data.json" if compression == "gz" else "data.json.gz")
        if other.exists():
            logger.warning(
                "Multiple raw snapshots found. Using %s and ignoring %s",
                fp,
                other,
            )

        n_seen += 1
        out_path = intermediate_full_history(intermediate_root, endpoint, ticker)

        if (not overwrite) and out_path.exists():
            logger.debug(
                "Skipping existing intermediate endpoint=%s ticker=%s path=%s",
                endpoint,
                ticker,
                out_path,
            )
            continue

        try:
            payload = load_payload(fp, compression=compression)
            df = payload_to_df(endpoint, payload)
            df = apply_endpoint_schema(endpoint, df)
            df = remove_duplicates(df, endpoint=endpoint,
                                   context=f"ticker={ticker}")
            n_read += 1

            if df.height == 0:
                logger.debug(
                    "Empty data endpoint=%s ticker=%s file=%s",
                    endpoint,
                    ticker,
                    fp,
                )
                write_parquet_atomic(
                    df,
                    out_path,
                    compression=parquet_compression,
                )
                out_paths.append(out_path)
                continue

            n_rows += df.height
            write_parquet_atomic(
                df,
                out_path,
                compression=parquet_compression,
            )
            out_paths.append(out_path)

        except Exception as e:
            failed.append(fp)

            if is_non_fatal_extract_error(e):
                logger.exception(
                    "Non-fatal extract error endpoint=%s ticker=%s file=%s",
                    endpoint,
                    ticker,
                    fp,
                )
                continue

            logger.exception(
                "Fatal extract error endpoint=%s ticker=%s file=%s",
                endpoint,
                ticker,
                fp,
            )
            raise

        if (n_seen % 5) == 0:
            logger.info(
                "Progress FULL_HISTORY endpoint=%s seen=%d "
                "read=%d written=%d failed=%d rows=%d",
                endpoint,
                n_seen,
                n_read,
                len(out_paths),
                len(failed),
                n_rows,
            )

    result = ExtractApiResult(
        endpoint=endpoint,
        strategy=DownloadStrategy.FULL_HISTORY,
        n_raw_files_seen=n_seen,
        n_raw_files_read=n_read,
        n_failed=len(failed),
        n_rows_total=n_rows,
        n_out_files=len(out_paths),
        duration_s=time.perf_counter() - t0,
        out_paths=out_paths,
        failed_paths=failed,
    )

    logger.info(
        "Finished extract FULL_HISTORY endpoint=%s seen=%d read=%d "
        "written=%d failed=%d rows=%d duration_s=%.2f",
        result.endpoint,
        result.n_raw_files_seen,
        result.n_raw_files_read,
        result.n_out_files,
        result.n_failed,
        result.n_rows_total,
        result.duration_s,
    )

    return result


def extract_by_trade_date(
    *,
    endpoint: str,
    raw_root: Path,
    intermediate_root: Path,
    tickers: list[str] | None,
    years: list[int],
    compression: str,
    overwrite: bool,
    parquet_compression: str,
) -> ExtractApiResult:
    """Extract BY_TRADE_DATE raw payloads and write one parquet per ticker."""
    t0 = time.perf_counter()
    out_paths: list[Path] = []
    failed: list[Path] = []
    n_seen = 0
    n_read = 0
    n_rows = 0

    suffix = glob_raw_suffix(compression)

    logger.info(
        "Extract BY_TRADE_DATE endpoint=%s years=%d",
        endpoint,
        len(years),
    )

    per_ticker: dict[str, list[pl.DataFrame]] = {}

    for year in years:
        year_t0 = time.perf_counter()
        year_dir = raw_by_trade_date_dir(raw_root, endpoint, year)
        files = sorted(year_dir.glob(suffix))
        if not files:
            logger.debug(
                "No raw files for endpoint=%s year=%d dir=%s",
                endpoint,
                year,
                year_dir,
            )
            continue

        logger.info(
            "Year %d: found %d raw files for endpoint=%s",
            year,
            len(files),
            endpoint,
        )

        for fp in files:
            n_seen += 1
            try:
                payload = load_payload(fp, compression=compression)
                df = payload_to_df(endpoint, payload)
                n_read += 1

                if df.height == 0:
                    continue

                if "ticker" not in df.columns:
                    raise KeyError(f"Missing 'ticker' column in {fp}")

                if tickers is not None:
                    df = df.filter(pl.col("ticker").is_in(tickers))
                    if df.height == 0:
                        continue

                for key, g in df.group_by("ticker", maintain_order=True):
                    t_val = key[0] if isinstance(key, tuple) else key
                    t_str = str(t_val)
                    per_ticker.setdefault(t_str, []).append(g)

                n_rows += df.height

            except Exception as e:
                failed.append(fp)

                if is_non_fatal_extract_error(e):
                    logger.exception(
                        "Non-fatal extract error endpoint=%s file=%s",
                        endpoint,
                        fp,
                    )
                    continue

                logger.exception(
                    "Fatal extract error endpoint=%s file=%s",
                    endpoint,
                    fp,
                )
                raise

        logger.info(
            "Year %d done endpoint=%s files=%d seen=%d read=%d "
            "failed=%d rows=%d tickers=%d duration_s=%.2f",
            year,
            endpoint,
            len(files),
            n_seen,
            n_read,
            len(failed),
            n_rows,
            len(per_ticker),
            time.perf_counter() - year_t0,
        )

    for t, dfs in per_ticker.items():
        out_path = intermediate_full_history(intermediate_root, endpoint, t)

        if (not overwrite) and out_path.exists():
            logger.debug(
                "Skipping existing intermediate endpoint=%s ticker=%s path=%s",
                endpoint,
                t,
                out_path,
            )
            continue

        df_all = pl.concat(dfs, how="diagonal_relaxed")
        df_all = apply_endpoint_schema(endpoint, df_all)
        df_all = remove_duplicates(df_all, endpoint=endpoint,
                                   context=f"ticker={t}")
        write_parquet_atomic(
            df_all,
            out_path,
            compression=parquet_compression,
        )
        out_paths.append(out_path)

    result = ExtractApiResult(
        endpoint=endpoint,
        strategy=DownloadStrategy.BY_TRADE_DATE,
        n_raw_files_seen=n_seen,
        n_raw_files_read=n_read,
        n_failed=len(failed),
        n_rows_total=n_rows,
        n_out_files=len(out_paths),
        duration_s=time.perf_counter() - t0,
        out_paths=out_paths,
        failed_paths=failed,
    )

    logger.info(
        "Finished extract BY_TRADE_DATE endpoint=%s years=%d seen=%d "
        "read=%d written=%d failed=%d rows=%d duration_s=%.2f",
        result.endpoint,
        len(years),
        result.n_raw_files_seen,
        result.n_raw_files_read,
        result.n_out_files,
        result.n_failed,
        result.n_rows_total,
        result.duration_s,
    )

    return result
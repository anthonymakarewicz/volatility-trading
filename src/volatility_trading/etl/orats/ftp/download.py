from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from ftplib import FTP
from pathlib import Path

from ._download_helpers import YearDownloadResult, download_one_year
from .types import DownloadFtpResult

logger = logging.getLogger(__name__)

DEFAULT_HOST: str = "orats.hostedftp.com"
DEFAULT_REMOTE_BASE_DIRS: tuple[str, ...] = (
    "smvstrikes_2007_2012",  # 2007–2012
    "smvstrikes",            # 2013–present
)


def download(
    *,
    user: str,
    password: str,
    raw_root: str | Path,
    host: str = DEFAULT_HOST,
    remote_base_dirs: Iterable[str] = DEFAULT_REMOTE_BASE_DIRS,
    year_whitelist: Iterable[int] | Iterable[str] | None = None,
    validate_zip: bool = True,
    max_workers: int = 1,
) -> DownloadFtpResult:
    """Download ORATS raw ZIP files from HostedFTP into a local directory.

    Remote layout (example)
    -----------------------
    smvstrikes_2007_2012/
        2007/
            ...
        2012/
            ...
    smvstrikes/
        2013/
            ...
        2025/
            ...

    Local layout (mirrored)
    -----------------------
    <raw_root>/
        smvstrikes_2007_2012/
            2007/*.zip
            ...
        smvstrikes/
            2013/*.zip
            ...

    Parameters
    ----------
    user:
        FTP username.
    password:
        FTP password.
    raw_root:
        Local root directory where ZIPs will be stored.
    host:
        FTP host to connect to. Defaults to `DEFAULT_HOST`.
    remote_base_dirs:
        Remote base directories to scan for yearly folders.
        Defaults to `DEFAULT_REMOTE_BASE_DIRS`.
    year_whitelist:
        Optional allowlist of years to download (e.g. `[2020, 2021]`). If None,
        downloads all available years under `remote_base_dirs`.
    validate_zip:
        If True, validate downloaded (and existing) files as ZIPs.
    max_workers:
        Number of parallel year-jobs to run. Use 1 for sequential downloads.

    Returns
    -------
    DownloadFtpResult
        Summary including counts, written paths, failures, and elapsed time.
    """
    t0 = time.perf_counter()
    raw_root_p = Path(raw_root)

    logger.info("Connecting to FTP host %s to discover jobs...", host)

    ftp = FTP(host)
    ftp.login(user, password)

    jobs: list[tuple[str, str]] = []

    years_allow: set[str] | None = None
    if year_whitelist is not None:
        years_allow = {str(y) for y in year_whitelist}

    try:
        for base in remote_base_dirs:
            logger.info("Discovering years in base directory: %s", base)

            ftp.cwd("/")
            ftp.cwd(base)

            years = sorted(ftp.nlst())
            logger.debug("Years found in %s: %s", base, years)

            for year_name in years:
                if not year_name.isdigit():
                    continue

                if years_allow is not None and str(year_name) not in years_allow:
                    continue

                jobs.append((base, year_name))

    finally:
        try:
            ftp.quit()
        except Exception:
            pass

    if not jobs:
        logger.info("No (base, year) jobs to download. Exiting.")
        return DownloadFtpResult(
            host=host,
            n_jobs=0,
            n_files_total=0,
            n_written=0,
            n_skipped=0,
            n_failed=0,
            duration_s=time.perf_counter() - t0,
            out_paths=[],
            failed_paths=[],
        )

    logger.info("Jobs to download: %d (max_workers=%d)", len(jobs), max_workers)

    year_results: list[YearDownloadResult] = []

    # Sequential path
    if max_workers <= 1:
        for base, year_name in jobs:
            year_results.append(
                download_one_year(
                    host=host,
                    user=user,
                    password=password,
                    base=base,
                    year_name=year_name,
                    raw_root=raw_root_p,
                    validate_zip=validate_zip,
                )
            )

    # Threaded path
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    download_one_year,
                    host=host,
                    user=user,
                    password=password,
                    base=base,
                    year_name=year_name,
                    raw_root=raw_root_p,
                    validate_zip=validate_zip,
                )
                for (base, year_name) in jobs
            ]

            for fut in futures:
                year_results.append(fut.result())

    # Aggregate
    out_paths: list[Path] = []
    failed_paths: list[Path] = []
    n_files_total = 0
    n_written = 0
    n_skipped = 0
    n_failed = 0

    for r in year_results:
        n_files_total += r.n_files_total
        n_written += r.n_written
        n_skipped += r.n_skipped
        n_failed += r.n_failed
        out_paths.extend(r.out_paths)
        failed_paths.extend(r.failed_paths)

    result = DownloadFtpResult(
        host=host,
        n_jobs=len(jobs),
        n_files_total=n_files_total,
        n_written=n_written,
        n_skipped=n_skipped,
        n_failed=n_failed,
        duration_s=time.perf_counter() - t0,
        out_paths=out_paths,
        failed_paths=failed_paths,
    )

    logger.info(
        "Finished FTP download host=%s jobs=%d files=%d "
        "written=%d skipped=%d failed=%d duration_s=%.2f",
        result.host,
        result.n_jobs,
        result.n_files_total,
        result.n_written,
        result.n_skipped,
        result.n_failed,
        result.duration_s,
    )

    return result
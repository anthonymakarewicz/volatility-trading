"""Result dataclasses for ORATS FTP download/extract runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DownloadFtpResult:
    """Summary metrics and paths produced by one FTP download run."""

    host: str
    n_jobs: int

    n_files_total: int
    n_written: int
    n_skipped: int
    n_failed: int

    duration_s: float

    out_paths: list[Path]
    failed_paths: list[Path]


@dataclass(frozen=True)
class ExtractFtpResult:
    """Summary metrics and paths produced by one FTP extraction run."""

    n_zip_files_seen: int
    n_zip_files_read: int
    n_zip_files_failed: int

    n_rows_total_before_dedup: int
    n_rows_total_after_dedup: int
    n_duplicates_dropped: int

    n_out_files: int
    duration_s: float

    out_paths: list[Path]
    failed_paths: list[Path]

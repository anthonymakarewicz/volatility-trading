from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .endpoints import DownloadStrategy


@dataclass(frozen=True)
class DownloadApiResult:
    """Summary of a download run."""

    endpoint: str
    strategy: DownloadStrategy
    n_requests_total: int
    n_written: int
    n_skipped: int
    n_empty_payloads: int
    n_failed: int
    duration_s: float
    out_paths: list[Path]
    failed_paths: list[Path]


@dataclass(frozen=True)
class ExtractApiResult:
    """Summary of an extraction run."""

    endpoint: str
    strategy: DownloadStrategy
    n_raw_files_seen: int
    n_raw_files_read: int
    n_failed: int
    n_rows_total: int
    n_out_files: int
    duration_s: float
    out_paths: list[Path]
    failed_paths: list[Path]

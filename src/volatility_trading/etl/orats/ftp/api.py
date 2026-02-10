"""Compatibility facade for the ORATS FTP ETL API.

This module re-exports the FTP download/extract entrypoints and result types.
"""

from __future__ import annotations

from .download.run import download
from .extract.run import extract
from .types import DownloadFtpResult, ExtractFtpResult

__all__ = [
    "DownloadFtpResult",
    "ExtractFtpResult",
    "download",
    "extract",
]

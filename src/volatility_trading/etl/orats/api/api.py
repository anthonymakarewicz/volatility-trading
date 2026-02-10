"""Compatibility facade for ORATS API ETL entrypoints and result types."""

from __future__ import annotations

from .download.run import download
from .extract.run import extract
from .types import DownloadApiResult, ExtractApiResult

__all__ = [
    "DownloadApiResult",
    "ExtractApiResult",
    "download",
    "extract",
]

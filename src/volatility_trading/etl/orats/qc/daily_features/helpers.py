"""
Internal helpers for the ORATS daily-features QC runner.
"""
from __future__ import annotations

import polars as pl

from ..common_helpers import run_all_checks as _run_all_checks
from ..types import QCCheckResult, QCConfig
from .specs import get_hard_specs, get_soft_specs



"""Shared internal serialization/summarization helpers for runner workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .workflow_types import BacktestWorkflowSpec

if TYPE_CHECKING:
    from .assembly import ResolvedWorkflowInputs


def serialize_runner_value(value: Any) -> Any:
    """Convert workflow/runtime objects into JSON-friendly structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        first = value.index.min() if not value.empty else None
        last = value.index.max() if not value.empty else None
        return {
            "type": "series",
            "name": value.name,
            "rows": int(len(value)),
            "start": None if first is None else pd.Timestamp(first).isoformat(),
            "end": None if last is None else pd.Timestamp(last).isoformat(),
        }
    if isinstance(value, Mapping):
        return {str(key): serialize_runner_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_runner_value(item) for item in value]
    if is_dataclass(value):
        return {
            field.name: serialize_runner_value(getattr(value, field.name))
            for field in fields(value)
        }
    return type(value).__name__


def build_resolved_summary(resolved: "ResolvedWorkflowInputs") -> dict[str, Any]:
    """Build a stable summary of resolved runtime inputs."""
    return {
        "strategy_name": resolved.strategy.name,
        "benchmark_name": resolved.benchmark_name,
        "risk_free_rate": serialize_runner_value(resolved.risk_free_rate),
    }


def build_report_config_payload(
    workflow: BacktestWorkflowSpec,
    resolved: "ResolvedWorkflowInputs",
) -> dict[str, Any]:
    """Build the persisted report payload for one workflow run."""
    return {
        "workflow": serialize_runner_value(workflow),
        "resolved": build_resolved_summary(resolved),
    }


def build_dry_run_plan(
    workflow: BacktestWorkflowSpec,
    resolved: "ResolvedWorkflowInputs",
) -> dict[str, Any]:
    """Build one stable dry-run summary payload."""
    return {
        "action": "backtest_run",
        "workflow": serialize_runner_value(workflow),
        "resolved": build_resolved_summary(resolved),
    }

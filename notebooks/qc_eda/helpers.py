"""QC summary helper utilities for the ORATS SPY QC EDA notebook."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
import polars as pl


class QCSummaryHelper:
    """Convenience wrapper around `qc_summary.json` rows keyed by check name."""

    def __init__(self, qc_summary: Sequence[Mapping[str, Any]]) -> None:
        self.qc_by_name: dict[str, dict[str, Any]] = {
            str(row["name"]): dict(row) for row in qc_summary if "name" in row
        }

    def qc_table(self, names: list[str]) -> pl.DataFrame:
        """Return a compact QC table for check names that exist."""
        rows: list[dict[str, Any]] = []
        for name in names:
            row = self.qc_by_name.get(name)
            if row is None:
                continue
            rows.append(
                {
                    "name": row["name"],
                    "severity": row["severity"],
                    "grade": row["grade"],
                    "passed": row["passed"],
                    "n_rows": row.get("n_rows"),
                    "n_units": row.get("n_units"),
                    "n_viol": row.get("n_viol"),
                    "viol_rate": row.get("viol_rate"),
                }
            )
        return pl.DataFrame(rows).sort(["severity", "name"])

    def qc_top_buckets(self, names: str | list[str]) -> pl.DataFrame:
        """Return top bucket diagnostics for one or many SOFT checks."""
        if isinstance(names, str):
            details = self._details_for(names)
            top_buckets = details.get("top_buckets", [])
            if not isinstance(top_buckets, list):
                top_buckets = []
            return pl.DataFrame(top_buckets)

        rows: list[dict[str, Any]] = []
        for name in names:
            details = self._details_for(name)
            top_buckets = details.get("top_buckets", [])
            if not isinstance(top_buckets, list):
                continue
            for bucket in top_buckets:
                if isinstance(bucket, Mapping):
                    rows.append({"name": name, **dict(bucket)})
        return pl.DataFrame(rows)

    def qc_thresholds(self, names: str | list[str]) -> dict[str, Any] | pl.DataFrame:
        """Return thresholds for one or many QC check names."""
        if isinstance(names, str):
            thresholds = self._details_for(names).get("thresholds", {})
            if isinstance(thresholds, Mapping):
                return dict(thresholds)
            return {}

        rows: list[dict[str, Any]] = []
        for name in names:
            thresholds = self._details_for(name).get("thresholds", {})
            if isinstance(thresholds, Mapping) and thresholds:
                rows.append({"name": name, **dict(thresholds)})
        return pl.DataFrame(rows).sort("name") if rows else pl.DataFrame()

    def qc_details(
        self, names: str | list[str]
    ) -> dict[str, Any] | dict[str, dict[str, Any]]:
        """Return details payload for one QC check or a name->details mapping."""
        if isinstance(names, str):
            return self._details_for(names)
        return {name: self._details_for(name) for name in names}

    def first_existing(self, *candidates: str) -> str | None:
        """Return first check name found in qc_summary."""
        for candidate in candidates:
            if candidate in self.qc_by_name:
                return candidate
        return None

    def info_stats_metric(self, info_name: str, metric: str) -> pd.DataFrame:
        """Return one metric block from INFO core_numeric_stats."""
        details = self._details_for(info_name)
        stats = details.get("stats", {})
        if not isinstance(stats, Mapping):
            return pd.DataFrame()
        if metric not in stats:
            return pd.DataFrame()
        metric_value = stats[metric]
        if not isinstance(metric_value, Mapping):
            return pd.DataFrame()
        out = pd.DataFrame([dict(metric_value)])
        out.insert(0, "metric", metric)
        return out

    def _details_for(self, name: str) -> dict[str, Any]:
        """Return a normalized details mapping for one QC check."""
        row = self.qc_by_name.get(name, {})
        details = row.get("details", {})
        if isinstance(details, Mapping):
            return dict(details)
        return {}

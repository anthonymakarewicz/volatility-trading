"""Sync FRED series to raw and processed parquet datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

try:
    from fredapi import Fred
except ImportError:  # pragma: no cover - optional dependency at runtime
    Fred = None


def _get_fred_client(*, token: str | None, token_env: str) -> Any:
    if Fred is None:
        raise RuntimeError(
            "fredapi is not installed. Install dependencies with requirements-dev.txt."
        )
    load_dotenv()
    resolved_token = token or os.getenv(token_env)
    if not resolved_token:
        raise RuntimeError(
            f"Missing FRED API key. Set env var {token_env} or pass token in config."
        )
    return Fred(api_key=resolved_token)


def _fetch_series(
    *,
    client: Any,
    series_id: str,
    alias: str,
    start: str | None,
    end: str | None,
) -> pd.Series:
    series = client.get_series(series_id, observation_start=start, observation_end=end)
    output = pd.Series(series, name=alias)
    output.index = pd.to_datetime(output.index)
    return output.sort_index()


def _write_raw_series(*, series: pd.Series, out_dir: Path, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{series.name}.parquet"
    if out_path.exists() and not overwrite:
        return
    series.to_frame(name="value").to_parquet(out_path, index=True)


def _domain_frame(
    *,
    client: Any,
    series_map: dict[str, str],
    start: str | None,
    end: str | None,
    raw_domain_root: Path,
    overwrite: bool,
) -> pd.DataFrame:
    columns: list[pd.Series] = []
    for alias, series_id in series_map.items():
        series = _fetch_series(
            client=client,
            series_id=series_id,
            alias=alias,
            start=start,
            end=end,
        )
        _write_raw_series(series=series, out_dir=raw_domain_root, overwrite=overwrite)
        columns.append(series)

    if not columns:
        return pd.DataFrame()
    return pd.concat(columns, axis=1).sort_index()


def sync_fred_domains(
    *,
    raw_root: Path,
    proc_root: Path,
    domains: dict[str, dict[str, str]],
    start: str | None = None,
    end: str | None = None,
    token: str | None = None,
    token_env: str = "FRED_API_KEY",
    asfreq_business_days: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Sync configured FRED domains and return processed parquet paths."""
    client = _get_fred_client(token=token, token_env=token_env)
    outputs: dict[str, Path] = {}

    for domain_name, series_map in domains.items():
        raw_domain_root = raw_root / domain_name
        proc_domain_root = proc_root / domain_name
        proc_domain_root.mkdir(parents=True, exist_ok=True)

        frame = _domain_frame(
            client=client,
            series_map=series_map,
            start=start,
            end=end,
            raw_domain_root=raw_domain_root,
            overwrite=overwrite,
        )
        if asfreq_business_days and not frame.empty:
            frame = frame.asfreq("B").ffill()

        proc_path = proc_domain_root / f"fred_{domain_name}.parquet"
        if proc_path.exists() and not overwrite:
            outputs[domain_name] = proc_path
            continue
        frame.to_parquet(proc_path, index=True)
        outputs[domain_name] = proc_path

    return outputs

"""Filesystem writers for backtest reporting bundles."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    BENCHMARK_COMPARISON_FILENAME,
    DASHBOARD_FILENAME,
    DEFAULT_REPORT_ROOT,
    EQUITY_DRAWDOWN_FILENAME,
    EXPOSURES_FILENAME,
    MANIFEST_FILENAME,
    MARGIN_DIAGNOSTICS_FILENAME,
    PLOTS_DIRNAME,
    PNL_ATTRIBUTION_FILENAME,
    ROLLING_METRICS_FILENAME,
    RUN_CONFIG_FILENAME,
    SUMMARY_METRICS_FILENAME,
    TRADES_FILENAME,
)
from .schemas import BacktestReportBundle


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "run"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _encode_trade_legs_csv_cell(value: Any) -> str:
    """Encode one ``trade_legs`` cell as canonical JSON string for CSV output."""
    return json.dumps(value, sort_keys=True, default=_json_default)


def _prepare_trades_for_csv(trades: pd.DataFrame) -> pd.DataFrame:
    """Prepare trades DataFrame for CSV persistence with stable ``trade_legs`` serialization."""
    out = trades.copy()
    if "trade_legs" not in out.columns:
        return out
    out["trade_legs"] = out["trade_legs"].map(_encode_trade_legs_csv_cell)
    return out


def write_report_bundle(
    bundle: BacktestReportBundle,
    *,
    output_root: Path = DEFAULT_REPORT_ROOT,
) -> Path:
    """Persist report bundle artifacts and return the run directory path."""
    strategy_slug = _sanitize_name(bundle.metadata.strategy_name)
    run_slug = _sanitize_name(bundle.metadata.run_id)
    run_dir = output_root / strategy_slug / run_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = run_dir / RUN_CONFIG_FILENAME
    summary_path = run_dir / SUMMARY_METRICS_FILENAME
    equity_path = run_dir / EQUITY_DRAWDOWN_FILENAME
    trades_path = run_dir / TRADES_FILENAME
    exposures_path = run_dir / EXPOSURES_FILENAME
    margin_path = run_dir / MARGIN_DIAGNOSTICS_FILENAME
    rolling_path = run_dir / ROLLING_METRICS_FILENAME
    pnl_attr_path = run_dir / PNL_ATTRIBUTION_FILENAME
    benchmark_comparison_path = run_dir / BENCHMARK_COMPARISON_FILENAME
    manifest_path = run_dir / MANIFEST_FILENAME

    _write_json(
        run_config_path,
        {"metadata": asdict(bundle.metadata), "config": bundle.run_config},
    )
    _write_json(summary_path, asdict(bundle.summary_metrics))
    bundle.equity_and_drawdown.to_csv(equity_path, index=True, index_label="date")
    _prepare_trades_for_csv(bundle.trades).to_csv(trades_path, index=False)
    bundle.exposures_daily.to_csv(exposures_path, index=True, index_label="date")
    bundle.margin_diagnostics_daily.to_csv(margin_path, index=True, index_label="date")
    bundle.rolling_metrics.to_csv(rolling_path, index=True, index_label="date")
    bundle.pnl_attribution_daily.to_csv(pnl_attr_path, index=True, index_label="date")
    if bundle.benchmark_comparison is not None:
        _write_json(benchmark_comparison_path, bundle.benchmark_comparison)

    plot_paths: dict[str, str] = {}
    if bundle.figures:
        plot_dir = run_dir / PLOTS_DIRNAME
        plot_dir.mkdir(parents=True, exist_ok=True)
        for name, figure in bundle.figures.items():
            filename = _sanitize_name(name) or DASHBOARD_FILENAME
            if not filename.endswith(".png"):
                filename = f"{filename}.png"
            file_path = plot_dir / filename
            figure.savefig(file_path, dpi=150, bbox_inches="tight")
            plot_paths[filename] = str(file_path.relative_to(run_dir))

    artifacts: dict[str, Any] = {
        RUN_CONFIG_FILENAME: RUN_CONFIG_FILENAME,
        SUMMARY_METRICS_FILENAME: SUMMARY_METRICS_FILENAME,
        EQUITY_DRAWDOWN_FILENAME: EQUITY_DRAWDOWN_FILENAME,
        TRADES_FILENAME: TRADES_FILENAME,
        EXPOSURES_FILENAME: EXPOSURES_FILENAME,
        MARGIN_DIAGNOSTICS_FILENAME: MARGIN_DIAGNOSTICS_FILENAME,
        ROLLING_METRICS_FILENAME: ROLLING_METRICS_FILENAME,
        PNL_ATTRIBUTION_FILENAME: PNL_ATTRIBUTION_FILENAME,
        "plots": plot_paths,
    }
    if bundle.benchmark_comparison is not None:
        artifacts[BENCHMARK_COMPARISON_FILENAME] = BENCHMARK_COMPARISON_FILENAME

    manifest = {
        "metadata": asdict(bundle.metadata),
        "artifacts": artifacts,
    }
    _write_json(manifest_path, manifest)

    return run_dir

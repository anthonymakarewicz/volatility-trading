from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class _DummyDF:
    columns: list[str]
    height: int


def test_build_daily_features_validates_inputs() -> None:
    mod = importlib.import_module(
        "volatility_trading.etl.orats.processed.daily_features.api"
    )

    with pytest.raises(ValueError, match="endpoints must be non-empty"):
        mod.build(
            inter_api_root="/tmp",
            proc_root="/tmp",
            ticker="SPX",
            endpoints=(),
        )

    with pytest.raises(ValueError, match="priority_endpoints must be a subset"):
        mod.build(
            inter_api_root="/tmp",
            proc_root="/tmp",
            ticker="SPX",
            endpoints=("summaries",),
            priority_endpoints=("hvs",),
        )


def test_build_daily_features_collect_stats_and_manifest_missing_endpoints(
    monkeypatch, tmp_path: Path
) -> None:
    mod = importlib.import_module(
        "volatility_trading.etl.orats.processed.daily_features.api"
    )

    # Only return one endpoint LF so we can validate missing_endpoints reporting.
    def _scan_inputs(*, inter_api_root: Path, ticker: str, endpoints, collect_stats: bool, stats_input_by_endpoint, **kwargs):
        assert collect_stats is True
        assert stats_input_by_endpoint is not None
        stats_input_by_endpoint["summaries"] = 10
        return {"summaries": "LF_SUM"}

    def _dedupe_endpoint(*, lf, endpoint: str, collect_stats: bool, stats_after_dedupe_by_endpoint, **kwargs):
        assert collect_stats is True
        assert stats_after_dedupe_by_endpoint is not None
        stats_after_dedupe_by_endpoint[endpoint] = 9
        return lf

    monkeypatch.setattr(mod.steps, "scan_inputs", _scan_inputs)
    monkeypatch.setattr(mod.steps, "dedupe_endpoint", _dedupe_endpoint)
    monkeypatch.setattr(mod.steps, "apply_bounds", lambda **kwargs: kwargs["lf"])

    def _build_spine(*, lfs: dict, endpoints, collect_stats: bool, stats_n_rows_spine, **kwargs):
        assert set(lfs.keys()) == {"summaries"}
        assert stats_n_rows_spine is not None
        stats_n_rows_spine.append(9)
        return "SPINE"

    monkeypatch.setattr(mod.steps, "build_key_spine", _build_spine)
    monkeypatch.setattr(mod.steps, "join_endpoints_on_spine", lambda **kwargs: "LF_JOINED")
    monkeypatch.setattr(mod.steps, "canonicalize_columns", lambda **kwargs: "LF_CANON")

    out_path = tmp_path / "underlying=SPX" / "part-0000.parquet"

    def _collect_and_write(*, columns, **kwargs):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"")
        return _DummyDF(columns=list(columns), height=7), out_path

    monkeypatch.setattr(mod.steps, "collect_and_write", _collect_and_write)

    captured: dict[str, object] = {}

    def _write_manifest_json(*, out_dir: Path, payload: dict):
        captured["payload"] = payload
        return out_dir / "manifest.json"

    monkeypatch.setattr(mod, "write_manifest_json", _write_manifest_json)

    result = mod.build(
        inter_api_root=tmp_path / "inter_api",
        proc_root=tmp_path / "proc",
        ticker="SPX",
        endpoints=("summaries", "hvs"),
        prefix_endpoint_cols=True,
        priority_endpoints=None,
        collect_stats=True,
        columns=("trade_date", "ticker"),
    )

    assert result.ticker == "SPX"
    assert result.out_path == out_path
    assert result.n_rows_written == 7
    assert result.n_rows_input_total == 10
    assert result.n_rows_spine == 9
    assert result.n_rows_input_by_endpoint == {"summaries": 10}
    assert result.n_rows_after_dedupe_by_endpoint == {"summaries": 9}

    payload = captured["payload"]
    assert payload["dataset"] == "orats_daily_features"
    assert payload["ticker"] == "SPX"
    assert payload["n_rows_written"] == 7
    assert payload["params"]["endpoints"] == ["summaries", "hvs"]
    assert payload["params"]["endpoints_used"] == ["summaries"]
    assert payload["params"]["missing_endpoints"] == ["hvs"]
    assert payload["params"]["prefix_endpoint_cols"] is True
    assert "built_at_utc" in payload


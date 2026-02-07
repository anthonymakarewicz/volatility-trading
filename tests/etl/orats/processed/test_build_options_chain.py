from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class _DummyDF:
    columns: list[str]
    height: int


def test_build_options_chain_collect_stats_and_writes_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    mod = importlib.import_module(
        "volatility_trading.etl.orats.processed.options_chain.api"
    )

    calls: list[tuple[str, object]] = []

    def _scan_inputs(*, collect_stats: bool, stats, **kwargs):
        calls.append(("scan_inputs", kwargs))
        if collect_stats:
            stats.n_rows_input = 100
        return "LF0"

    def _filter_preferred_opra_root(*, lf, ticker):
        calls.append(("filter_preferred_opra_root", ticker))
        assert lf == "LF0"
        return "LF1"

    def _dedupe(*, lf, collect_stats: bool, stats, **kwargs):
        calls.append(("dedupe_options_chain", kwargs))
        assert lf == "LF1"
        if collect_stats:
            stats.n_rows_after_dedupe = 90
        return "LF2"

    def _merge_yield(*, lf, collect_stats: bool, stats, merge_dividend_yield: bool, **kwargs):
        calls.append(("merge_dividend_yield", merge_dividend_yield))
        assert lf == "LF2"
        if collect_stats:
            stats.n_rows_yield_input = 90
            stats.n_rows_yield_after_dedupe = 90
            stats.n_rows_join_missing_yield = 3
        return "LF3"

    def _unify_spot(*, lf):
        calls.append(("unify_spot_price", None))
        assert lf == "LF3"
        return "LF4"

    def _apply_bounds(*, lf, **kwargs):
        calls.append(("apply_bounds", kwargs.get("ticker")))
        assert lf == "LF4"
        return "LF5"

    def _derived(*, lf):
        calls.append(("add_derived_features", None))
        assert lf == "LF5"
        return "LF6"

    def _filters(*, lf, collect_stats: bool, stats, dte_min: int, dte_max: int, **kwargs):
        calls.append(("apply_filters", (dte_min, dte_max)))
        assert lf == "LF6"
        if collect_stats:
            stats.n_rows_after_trading = 80
            stats.n_rows_after_hard = 70
        return "LF7"

    def _put_greeks(*, lf):
        calls.append(("add_put_greeks", None))
        assert lf == "LF7"
        return "LF8"

    def _put_greeks_simple(*, lf):
        pytest.fail("add_put_greeks_simple should not be called when derive_put_greeks=True")

    out_path = tmp_path / "underlying=SPX" / "part-0000.parquet"

    def _collect_and_write(*, lf, proc_root: Path, ticker: str, columns):
        calls.append(("collect_and_write", (proc_root, ticker, tuple(columns))))
        assert lf == "LF8"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"")
        return _DummyDF(columns=list(columns), height=123), out_path

    captured: dict[str, object] = {}

    def _write_manifest_json(*, out_dir: Path, payload: dict):
        captured["out_dir"] = out_dir
        captured["payload"] = payload
        p = out_dir / "manifest.json"
        p.write_text("{}", encoding="utf-8")
        return p

    monkeypatch.setattr(mod.steps, "scan_inputs", _scan_inputs)
    monkeypatch.setattr(mod.steps, "filter_preferred_opra_root", _filter_preferred_opra_root)
    monkeypatch.setattr(mod.steps, "dedupe_options_chain", _dedupe)
    monkeypatch.setattr(mod.steps, "merge_dividend_yield", _merge_yield)
    monkeypatch.setattr(mod.steps, "unify_spot_price", _unify_spot)
    monkeypatch.setattr(mod.steps, "apply_bounds", _apply_bounds)
    monkeypatch.setattr(mod.steps, "add_derived_features", _derived)
    monkeypatch.setattr(mod.steps, "apply_filters", _filters)
    monkeypatch.setattr(mod.steps, "add_put_greeks", _put_greeks)
    monkeypatch.setattr(mod.steps, "add_put_greeks_simple", _put_greeks_simple)
    monkeypatch.setattr(mod.steps, "collect_and_write", _collect_and_write)
    monkeypatch.setattr(mod, "write_manifest_json", _write_manifest_json)
    monkeypatch.setattr(mod, "OPTION_EXERCISE_STYLE", {"SPX": "EU"})

    result = mod.build(
        inter_root=tmp_path / "inter",
        proc_root=tmp_path / "proc",
        ticker="SPX",
        years=[2020, "2021"],
        dte_min=10,
        dte_max=30,
        moneyness_min=0.9,
        moneyness_max=1.1,
        monies_implied_inter_root=tmp_path / "api_inter",
        merge_dividend_yield=True,
        derive_put_greeks=True,
        collect_stats=True,
        columns=("a", "b"),
    )

    assert result.ticker == "SPX"
    assert result.out_path == out_path
    assert result.n_rows_written == 123
    assert result.n_rows_input == 100
    assert result.n_rows_after_dedupe == 90
    assert result.n_rows_yield_input == 90
    assert result.n_rows_yield_after_dedupe == 90
    assert result.n_rows_join_missing_yield == 3
    assert result.n_rows_after_trading == 80
    assert result.n_rows_after_hard == 70

    payload = captured["payload"]
    assert payload["dataset"] == "orats_options_chain"
    assert payload["ticker"] == "SPX"
    assert payload["n_rows_written"] == 123
    assert payload["columns"] == ["a", "b"]
    assert payload["params"]["put_greeks_mode"] == "parity"
    assert payload["params"]["exercise_style"] == "EU"
    assert payload["params"]["merge_dividend_yield"] is True
    assert payload["params"]["years"] == ["2020", "2021"]
    assert payload["params"]["dte_min"] == 10
    assert payload["params"]["dte_max"] == 30
    assert payload["params"]["moneyness_min"] == 0.9
    assert payload["params"]["moneyness_max"] == 1.1
    assert "built_at_utc" in payload


def test_build_options_chain_put_greeks_simple_mode(monkeypatch, tmp_path: Path) -> None:
    mod = importlib.import_module(
        "volatility_trading.etl.orats.processed.options_chain.api"
    )

    monkeypatch.setattr(mod.steps, "scan_inputs", lambda **kwargs: "LF0")
    monkeypatch.setattr(mod.steps, "filter_preferred_opra_root", lambda lf, ticker: lf)
    monkeypatch.setattr(mod.steps, "dedupe_options_chain", lambda **kwargs: "LF1")
    monkeypatch.setattr(mod.steps, "merge_dividend_yield", lambda **kwargs: "LF2")
    monkeypatch.setattr(mod.steps, "unify_spot_price", lambda lf: lf)
    monkeypatch.setattr(mod.steps, "apply_bounds", lambda **kwargs: "LF3")
    monkeypatch.setattr(mod.steps, "add_derived_features", lambda lf: lf)
    monkeypatch.setattr(mod.steps, "apply_filters", lambda **kwargs: "LF4")

    called = {"simple": 0}

    def _simple(*, lf):
        called["simple"] += 1
        return lf

    monkeypatch.setattr(mod.steps, "add_put_greeks_simple", _simple)
    monkeypatch.setattr(mod.steps, "add_put_greeks", lambda **kwargs: pytest.fail("should not call parity greeks"))
    monkeypatch.setattr(
        mod.steps,
        "collect_and_write",
        lambda **kwargs: (_DummyDF(columns=["x"], height=1), tmp_path / "out.parquet"),
    )
    monkeypatch.setattr(mod, "write_manifest_json", lambda **kwargs: tmp_path / "manifest.json")

    mod.build(
        inter_root=tmp_path / "inter",
        proc_root=tmp_path / "proc",
        ticker="SPX",
        monies_implied_inter_root=None,
        merge_dividend_yield=False,
        derive_put_greeks=False,
        collect_stats=False,
    )

    assert called["simple"] == 1


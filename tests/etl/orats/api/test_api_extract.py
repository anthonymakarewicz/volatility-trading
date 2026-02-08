from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass

import pytest

from volatility_trading.etl.orats.api.endpoints import DownloadStrategy, EndpointSpec


@dataclass(frozen=True)
class _DummyResult:
    n_failed: int = 0


def test_extract_full_history_ignores_years(monkeypatch, caplog) -> None:
    sys.modules.pop("extract", None)
    mod = importlib.import_module("volatility_trading.etl.orats.api.extract.run")

    spec = EndpointSpec(
        path="/dummy",
        strategy=DownloadStrategy.FULL_HISTORY,
        required=("ticker",),
    )
    monkeypatch.setattr(mod, "get_endpoint_spec", lambda endpoint: spec)

    captured: dict[str, object] = {}

    def handler(**kwargs):
        captured.update(kwargs)
        return _DummyResult()

    monkeypatch.setattr(mod, "extract_full_history", handler)

    result = mod.extract(
        endpoint="hvs",
        raw_root="/tmp",
        intermediate_root="/tmp",
        tickers=["SPX", " ", None, "SPX"],
        year_whitelist=[2020],
        compression="gz",
        overwrite=False,
        parquet_compression="zstd",
    )

    assert isinstance(result, _DummyResult)
    assert captured["tickers"] == ["SPX"]
    assert any("year_whitelist is ignored" in rec.message for rec in caplog.records)


def test_extract_by_trade_date_requires_years(monkeypatch) -> None:
    sys.modules.pop("extract", None)
    mod = importlib.import_module("volatility_trading.etl.orats.api.extract.run")

    spec = EndpointSpec(
        path="/dummy",
        strategy=DownloadStrategy.BY_TRADE_DATE,
        required=("ticker", "tradeDate"),
    )
    monkeypatch.setattr(mod, "get_endpoint_spec", lambda endpoint: spec)

    with pytest.raises(ValueError, match="year_whitelist must be provided"):
        mod.extract(
            endpoint="monies_implied",
            raw_root="/tmp",
            intermediate_root="/tmp",
            tickers=["SPX"],
            year_whitelist=None,
            compression="gz",
            overwrite=False,
            parquet_compression="zstd",
        )


def test_extract_by_trade_date_uses_validate_years(monkeypatch) -> None:
    sys.modules.pop("extract", None)
    mod = importlib.import_module("volatility_trading.etl.orats.api.extract.run")

    spec = EndpointSpec(
        path="/dummy",
        strategy=DownloadStrategy.BY_TRADE_DATE,
        required=("ticker", "tradeDate"),
    )
    monkeypatch.setattr(mod, "get_endpoint_spec", lambda endpoint: spec)
    monkeypatch.setattr(mod, "validate_years", lambda years: [2018, 2019])

    captured: dict[str, object] = {}

    def handler(**kwargs):
        captured.update(kwargs)
        return _DummyResult()

    monkeypatch.setattr(mod, "extract_by_trade_date", handler)

    mod.extract(
        endpoint="monies_implied",
        raw_root="/tmp",
        intermediate_root="/tmp",
        tickers=["SPX"],
        year_whitelist=[2018],
        compression="gz",
        overwrite=False,
        parquet_compression="zstd",
    )

    assert captured["years"] == [2018, 2019]


def test_extract_invalid_compression() -> None:
    sys.modules.pop("extract", None)
    mod = importlib.import_module("volatility_trading.etl.orats.api.extract.run")

    with pytest.raises(ValueError, match="Unsupported compression"):
        mod.extract(
            endpoint="hvs",
            raw_root="/tmp",
            intermediate_root="/tmp",
            tickers=["SPX"],
            compression="zip",
            overwrite=False,
            parquet_compression="zstd",
        )


def test_extract_empty_tickers_raises(monkeypatch) -> None:
    sys.modules.pop("extract", None)
    mod = importlib.import_module("volatility_trading.etl.orats.api.extract.run")

    spec = EndpointSpec(
        path="/dummy",
        strategy=DownloadStrategy.FULL_HISTORY,
        required=("ticker",),
    )
    monkeypatch.setattr(mod, "get_endpoint_spec", lambda endpoint: spec)

    with pytest.raises(ValueError, match="tickers is passed but none of them is valid"):
        mod.extract(
            endpoint="hvs",
            raw_root="/tmp",
            intermediate_root="/tmp",
            tickers=[" ", None],
            compression="gz",
            overwrite=False,
            parquet_compression="zstd",
        )

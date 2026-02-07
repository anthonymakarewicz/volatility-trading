from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys

import pytest

from volatility_trading.etl.orats.api.endpoints import DownloadStrategy, EndpointSpec


@dataclass(frozen=True)
class _DummyResult:
    n_failed: int = 0


class _DummySession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_download_full_history_ignores_years(monkeypatch, caplog) -> None:
    sys.modules.pop("download", None)
    mod = importlib.import_module("volatility_trading.etl.orats.api.download.run")

    spec = EndpointSpec(
        path="/dummy",
        strategy=DownloadStrategy.FULL_HISTORY,
        required=("ticker",),
    )
    monkeypatch.setattr(mod, "get_endpoint_spec", lambda endpoint: spec)
    monkeypatch.setattr(mod, "OratsClient", lambda token: object())
    monkeypatch.setattr(mod.requests, "Session", lambda: _DummySession())

    captured: dict[str, object] = {}

    def handler(**kwargs):
        captured.update(kwargs)
        return _DummyResult()

    monkeypatch.setattr(
        mod,
        "DOWNLOAD_HANDLERS",
        {DownloadStrategy.FULL_HISTORY: handler},
    )

    result = mod.download(
        token="t",
        endpoint="ivrank",
        raw_root="/tmp",
        tickers=["SPX", " ", None, "SPX"],
        year_whitelist=[2020],
        fields=None,
        compression="gz",
        sleep_s=0.0,
        overwrite=False,
    )

    assert isinstance(result, _DummyResult)
    assert captured["tickers"] == ["SPX"]
    assert "years" not in captured
    assert any(
        "year_whitelist is ignored" in rec.message for rec in caplog.records
    )


def test_download_by_trade_date_requires_years(monkeypatch) -> None:
    sys.modules.pop("download", None)
    mod = importlib.import_module("volatility_trading.etl.orats.api.download.run")

    spec = EndpointSpec(
        path="/dummy",
        strategy=DownloadStrategy.BY_TRADE_DATE,
        required=("ticker", "tradeDate"),
    )
    monkeypatch.setattr(mod, "get_endpoint_spec", lambda endpoint: spec)
    monkeypatch.setattr(mod, "OratsClient", lambda token: object())

    with pytest.raises(ValueError, match="year_whitelist must be provided"):
        mod.download(
            token="t",
            endpoint="monies_implied",
            raw_root="/tmp",
            tickers=["SPX"],
            year_whitelist=None,
            fields=None,
            compression="gz",
        )


def test_download_by_trade_date_uses_validate_years(monkeypatch) -> None:
    sys.modules.pop("download", None)
    mod = importlib.import_module("volatility_trading.etl.orats.api.download.run")

    spec = EndpointSpec(
        path="/dummy",
        strategy=DownloadStrategy.BY_TRADE_DATE,
        required=("ticker", "tradeDate"),
    )
    monkeypatch.setattr(mod, "get_endpoint_spec", lambda endpoint: spec)
    monkeypatch.setattr(mod, "OratsClient", lambda token: object())
    monkeypatch.setattr(mod.requests, "Session", lambda: _DummySession())
    monkeypatch.setattr(mod, "validate_years", lambda years: [2019, 2020])

    captured: dict[str, object] = {}

    def handler(**kwargs):
        captured.update(kwargs)
        return _DummyResult()

    monkeypatch.setattr(
        mod,
        "DOWNLOAD_HANDLERS",
        {DownloadStrategy.BY_TRADE_DATE: handler},
    )

    mod.download(
        token="t",
        endpoint="monies_implied",
        raw_root="/tmp",
        tickers=["SPX"],
        year_whitelist=[2019],
        fields=None,
        compression="gz",
    )

    assert captured["years"] == [2019, 2020]


def test_download_invalid_compression() -> None:
    sys.modules.pop("download", None)
    mod = importlib.import_module("volatility_trading.etl.orats.api.download.run")

    with pytest.raises(ValueError, match="Unsupported compression"):
        mod.download(
            token="t",
            endpoint="ivrank",
            raw_root="/tmp",
            tickers=["SPX"],
            compression="zip",
        )


def test_download_empty_tickers_raises(monkeypatch) -> None:
    sys.modules.pop("download", None)
    mod = importlib.import_module("volatility_trading.etl.orats.api.download.run")

    spec = EndpointSpec(
        path="/dummy",
        strategy=DownloadStrategy.FULL_HISTORY,
        required=("ticker",),
    )
    monkeypatch.setattr(mod, "get_endpoint_spec", lambda endpoint: spec)

    with pytest.raises(ValueError, match="tickers must be non-empty"):
        mod.download(
            token="t",
            endpoint="ivrank",
            raw_root="/tmp",
            tickers=[" ", None],
            compression="gz",
        )

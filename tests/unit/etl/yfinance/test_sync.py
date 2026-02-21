from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from volatility_trading.etl.yfinance.sync import sync_yfinance_time_series


def _sample_frame(value: float) -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    return pd.DataFrame(
        {
            "open": [value, value + 1.0],
            "high": [value + 2.0, value + 3.0],
            "low": [value - 1.0, value],
            "close": [value + 0.5, value + 1.5],
            "adj_close": [value + 0.4, value + 1.4],
            "volume": [1000, 1200],
        },
        index=index,
    )


def test_sync_uses_internal_download_symbols_and_clean_storage_tickers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    frames = {
        "^SP500TR": _sample_frame(5000.0),
        "^VIX": _sample_frame(20.0),
        "SPY": _sample_frame(480.0),
    }

    def _fake_download_ticker(
        *,
        ticker: str,
        start: str | None,
        end: str | None,
        interval: str,
        auto_adjust: bool,
        actions: bool,
    ) -> pd.DataFrame:
        _ = (start, end, interval, auto_adjust, actions)
        calls.append(ticker)
        return frames[ticker]

    monkeypatch.setattr(
        "volatility_trading.etl.yfinance.sync._download_ticker",
        _fake_download_ticker,
    )

    raw_root = tmp_path / "raw"
    proc_root = tmp_path / "processed"

    out_path = sync_yfinance_time_series(
        tickers=["SP500TR", "VIX", "SPY"],
        raw_root=raw_root,
        proc_root=proc_root,
        overwrite=True,
    )

    assert calls == ["^SP500TR", "^VIX", "SPY"]
    assert (raw_root / "SP500TR.parquet").exists()
    assert (raw_root / "VIX.parquet").exists()
    assert (raw_root / "SPY.parquet").exists()
    assert out_path == proc_root / "yfinance_time_series.parquet"

    processed = pd.read_parquet(out_path)
    assert set(processed["ticker"].unique()) == {"SP500TR", "VIX", "SPY"}


def test_sync_rejects_ambiguous_storage_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Ambiguous ticker mapping"):
        sync_yfinance_time_series(
            tickers=["ABC", "^ABC"],
            raw_root=tmp_path / "raw",
            proc_root=tmp_path / "processed",
            overwrite=True,
        )

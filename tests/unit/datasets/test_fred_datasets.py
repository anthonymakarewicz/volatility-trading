from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from volatility_trading.datasets.fred import (
    fred_domain_path,
    fred_market_path,
    fred_rates_path,
    read_fred_domain,
    read_fred_market,
    read_fred_rates,
    scan_fred_domain,
)


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def test_fred_domain_path_builds_expected_location(tmp_path: Path) -> None:
    path = fred_domain_path("rates", proc_root=tmp_path)
    assert path == tmp_path / "rates" / "fred_rates.parquet"


def test_fred_domain_path_rejects_empty_domain(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="domain must be non-empty"):
        fred_domain_path("  ", proc_root=tmp_path)


def test_scan_fred_domain_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(
        FileNotFoundError, match="Processed FRED domain dataset not found"
    ):
        scan_fred_domain("rates", proc_root=tmp_path).collect()


def test_read_fred_domain_normalizes_index_column_name(tmp_path: Path) -> None:
    path = tmp_path / "rates" / "fred_rates.parquet"
    _write_parquet(
        path,
        pl.DataFrame(
            {
                "__index_level_0__": [dt.date(2020, 1, 1), dt.date(2020, 1, 2)],
                "dgs3mo": [0.015, 0.016],
                "dgs10": [0.018, 0.019],
            }
        ),
    )

    out = read_fred_domain("rates", proc_root=tmp_path, columns=["date", "dgs3mo"])

    assert out.columns == ["date", "dgs3mo"]
    assert out["date"].to_list() == [dt.date(2020, 1, 1), dt.date(2020, 1, 2)]
    assert out["dgs3mo"].to_list() == [0.015, 0.016]


def test_read_fred_rates_and_market_from_domain_roots(tmp_path: Path) -> None:
    rates_root = tmp_path / "rates"
    market_root = tmp_path / "market"
    _write_parquet(
        fred_rates_path(rates_root),
        pl.DataFrame(
            {
                "__index_level_0__": [dt.date(2021, 1, 4)],
                "dgs3mo": [0.0025],
            }
        ),
    )
    _write_parquet(
        fred_market_path(market_root),
        pl.DataFrame(
            {
                "__index_level_0__": [dt.date(2021, 1, 4)],
                "vixcls": [24.3],
            }
        ),
    )

    rates = read_fred_rates(proc_root=rates_root)
    market = read_fred_market(proc_root=market_root)

    assert rates.columns == ["date", "dgs3mo"]
    assert market.columns == ["date", "vixcls"]

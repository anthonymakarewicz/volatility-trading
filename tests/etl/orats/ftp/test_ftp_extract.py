from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _make_year_dir(raw_root: Path, year: str, base: str = "smvstrikes") -> Path:
    year_dir = raw_root / base / year
    year_dir.mkdir(parents=True)
    return year_dir


def test_ftp_extract_respects_year_whitelist_and_writes_partitions(
    monkeypatch, ftp_roots, write_zip_files, write_parquet_stub
) -> None:
    mod = importlib.import_module("volatility_trading.etl.orats.ftp.extract.run")

    raw_root, out_root = ftp_roots
    year_2020 = _make_year_dir(raw_root, "2020")
    year_2021 = _make_year_dir(raw_root, "2021")
    write_zip_files(year_2020, ["a.zip", "b.zip"])
    write_zip_files(year_2021, ["c.zip"])

    def _read_zip(path: Path):
        # Include both tickers so we get one output parquet per ticker.
        return mod.pl.DataFrame(
            {
                mod.ROOT_COL: ["SPX", "SPX", "AAPL"],
                "x": [1, 1, 2],
            }
        )

    monkeypatch.setattr(mod, "read_orats_zip_to_polars", _read_zip)

    monkeypatch.setattr(mod.pl.DataFrame, "write_parquet", write_parquet_stub)

    result = mod.extract(
        raw_root=raw_root,
        out_root=out_root,
        tickers=["SPX", "AAPL"],
        year_whitelist=[2020],
        strict=True,
    )

    assert result.n_zip_files_seen == 2
    assert result.n_zip_files_read == 2
    assert result.n_zip_files_failed == 0
    assert result.n_out_files == 2

    spx_out = out_root / "underlying=SPX" / "year=2020" / "part-0000.parquet"
    aapl_out = out_root / "underlying=AAPL" / "year=2020" / "part-0000.parquet"
    assert spx_out.exists()
    assert aapl_out.exists()

    # Make sure we truly skipped other years.
    assert not (out_root / "underlying=SPX" / "year=2021").exists()
    assert not (out_root / "underlying=AAPL" / "year=2021").exists()


def test_ftp_extract_strict_raises_on_failed_zip(
    monkeypatch, ftp_roots, write_zip_files, write_parquet_stub
) -> None:
    mod = importlib.import_module("volatility_trading.etl.orats.ftp.extract.run")

    raw_root, out_root = ftp_roots
    year_2020 = _make_year_dir(raw_root, "2020")
    write_zip_files(year_2020, ["bad.zip"])

    def _read_zip_fail(path: Path):
        raise mod.NoDataError("no rows")

    monkeypatch.setattr(mod, "read_orats_zip_to_polars", _read_zip_fail)
    monkeypatch.setattr(mod.pl.DataFrame, "write_parquet", write_parquet_stub)

    with pytest.raises(RuntimeError, match="problematic ZIP files"):
        mod.extract(
            raw_root=raw_root,
            out_root=out_root,
            tickers=["SPX"],
            year_whitelist=[2020],
            strict=True,
        )


def test_ftp_extract_non_strict_records_failures(
    monkeypatch, ftp_roots, write_zip_files
) -> None:
    mod = importlib.import_module("volatility_trading.etl.orats.ftp.extract.run")

    raw_root, out_root = ftp_roots
    year_2020 = _make_year_dir(raw_root, "2020")
    bad = year_2020 / "bad.zip"
    write_zip_files(year_2020, [bad.name])

    def _read_zip_fail(path: Path):
        raise mod.NoDataError("no rows")

    monkeypatch.setattr(mod, "read_orats_zip_to_polars", _read_zip_fail)

    result = mod.extract(
        raw_root=raw_root,
        out_root=out_root,
        tickers=["SPX"],
        year_whitelist=[2020],
        strict=False,
    )

    assert result.n_zip_files_seen == 1
    assert result.n_zip_files_read == 0
    assert result.n_zip_files_failed == 1
    assert result.failed_paths == [bad]
    assert result.n_out_files == 0


def test_ftp_extract_deduplicates_exact_rows(
    monkeypatch, ftp_roots, write_zip_files, write_parquet_stub
) -> None:
    mod = importlib.import_module("volatility_trading.etl.orats.ftp.extract.run")

    raw_root, out_root = ftp_roots
    year_2020 = _make_year_dir(raw_root, "2020")
    write_zip_files(year_2020, ["a.zip", "b.zip"])

    def _read_zip(path: Path):
        return mod.pl.DataFrame({mod.ROOT_COL: ["SPX"], "x": [1]})

    monkeypatch.setattr(mod, "read_orats_zip_to_polars", _read_zip)

    monkeypatch.setattr(mod.pl.DataFrame, "write_parquet", write_parquet_stub)

    result = mod.extract(
        raw_root=raw_root,
        out_root=out_root,
        tickers=["SPX"],
        year_whitelist=[2020],
        strict=True,
    )

    assert result.n_zip_files_seen == 2
    assert result.n_zip_files_read == 2
    assert result.n_duplicates_dropped == 1
    assert result.n_rows_total_before_dedup == 2
    assert result.n_rows_total_after_dedup == 1
    assert result.n_out_files == 1

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def test_ftp_extract_respects_year_whitelist_and_writes_partitions(
    monkeypatch, tmp_path: Path
) -> None:
    mod = importlib.import_module("volatility_trading.etl.orats.ftp.extract.run")

    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    year_2020 = raw_root / "smvstrikes" / "2020"
    year_2021 = raw_root / "smvstrikes" / "2021"
    year_2020.mkdir(parents=True)
    year_2021.mkdir(parents=True)

    (year_2020 / "a.zip").write_text("", encoding="utf-8")
    (year_2020 / "b.zip").write_text("", encoding="utf-8")
    (year_2021 / "c.zip").write_text("", encoding="utf-8")

    def _read_zip(path: Path):
        # Include both tickers so we get one output parquet per ticker.
        return mod.pl.DataFrame(
            {
                mod.ROOT_COL: ["SPX", "SPX", "AAPL"],
                "x": [1, 1, 2],
            }
        )

    monkeypatch.setattr(mod, "read_orats_zip_to_polars", _read_zip)

    def _fake_write_parquet(self, file, *args, **kwargs) -> None:
        p = Path(file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")

    monkeypatch.setattr(mod.pl.DataFrame, "write_parquet", _fake_write_parquet)

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


def test_ftp_extract_strict_raises_on_failed_zip(monkeypatch, tmp_path: Path) -> None:
    mod = importlib.import_module("volatility_trading.etl.orats.ftp.extract.run")

    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    year_2020 = raw_root / "smvstrikes" / "2020"
    year_2020.mkdir(parents=True)
    (year_2020 / "bad.zip").write_text("", encoding="utf-8")

    def _read_zip_fail(path: Path):
        raise mod.NoDataError("no rows")

    monkeypatch.setattr(mod, "read_orats_zip_to_polars", _read_zip_fail)
    monkeypatch.setattr(
        mod.pl.DataFrame,
        "write_parquet",
        lambda self, file, *args, **kwargs: Path(file).write_bytes(b""),
    )

    with pytest.raises(RuntimeError, match="problematic ZIP files"):
        mod.extract(
            raw_root=raw_root,
            out_root=out_root,
            tickers=["SPX"],
            year_whitelist=[2020],
            strict=True,
        )


def test_ftp_extract_non_strict_records_failures(monkeypatch, tmp_path: Path) -> None:
    mod = importlib.import_module("volatility_trading.etl.orats.ftp.extract.run")

    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    year_2020 = raw_root / "smvstrikes" / "2020"
    year_2020.mkdir(parents=True)
    bad = year_2020 / "bad.zip"
    bad.write_text("", encoding="utf-8")

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


def test_ftp_extract_deduplicates_exact_rows(monkeypatch, tmp_path: Path) -> None:
    mod = importlib.import_module("volatility_trading.etl.orats.ftp.extract.run")

    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    year_2020 = raw_root / "smvstrikes" / "2020"
    year_2020.mkdir(parents=True)

    (year_2020 / "a.zip").write_text("", encoding="utf-8")
    (year_2020 / "b.zip").write_text("", encoding="utf-8")

    def _read_zip(path: Path):
        return mod.pl.DataFrame({mod.ROOT_COL: ["SPX"], "x": [1]})

    monkeypatch.setattr(mod, "read_orats_zip_to_polars", _read_zip)

    def _fake_write_parquet(self, file, *args, **kwargs) -> None:
        p = Path(file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")

    monkeypatch.setattr(mod.pl.DataFrame, "write_parquet", _fake_write_parquet)

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
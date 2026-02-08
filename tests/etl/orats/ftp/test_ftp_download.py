from __future__ import annotations

import importlib
from pathlib import Path


def _make_download_one_year(
    mod,
    *,
    calls: list[tuple[str, str, bool]] | None = None,
    result_factory=None,
):
    def _download_one_year(*, base: str, year_name: str, validate_zip: bool, **kwargs):
        if calls is not None:
            calls.append((base, year_name, validate_zip))
        if result_factory is not None:
            return result_factory(base, year_name)
        return mod.YearDownloadResult(
            base=base,
            year_name=year_name,
            n_files_total=3,
            n_written=2,
            n_skipped=1,
            n_failed=0,
            out_paths=[Path("/tmp/a.zip"), Path("/tmp/b.zip")],
            failed_paths=[],
        )

    return _download_one_year


def test_ftp_download_filters_year_whitelist_and_aggregates(
    monkeypatch,
    tmp_path: Path,
    dummy_ftp_factory,
) -> None:
    mod = importlib.import_module("volatility_trading.etl.orats.ftp.download.run")

    base_to_years = {"base": ["2019", "2020", "notes"]}
    monkeypatch.setattr(mod, "FTP", dummy_ftp_factory(base_to_years))

    calls: list[tuple[str, str, bool]] = []
    monkeypatch.setattr(
        mod, "download_one_year", _make_download_one_year(mod, calls=calls)
    )

    result = mod.download(
        user="u",
        password="p",
        raw_root=tmp_path,
        host="h",
        remote_base_dirs=["base"],
        year_whitelist=[2020],
        validate_zip=False,
        max_workers=1,
    )

    assert calls == [("base", "2020", False)]
    assert result.host == "h"
    assert result.n_jobs == 1
    assert result.n_files_total == 3
    assert result.n_written == 2
    assert result.n_skipped == 1
    assert result.n_failed == 0
    assert len(result.out_paths) == 2


def test_ftp_download_no_jobs_returns_empty(
    monkeypatch,
    tmp_path: Path,
    dummy_ftp_factory,
) -> None:
    mod = importlib.import_module("volatility_trading.etl.orats.ftp.download.run")

    base_to_years = {"base": ["README", "notes"]}
    instances: list[object] = []
    monkeypatch.setattr(
        mod,
        "FTP",
        dummy_ftp_factory(base_to_years, instances=instances, track_login=False),
    )
    monkeypatch.setattr(mod, "download_one_year", lambda **kwargs: None)

    result = mod.download(
        user="u",
        password="p",
        raw_root=tmp_path,
        host="h",
        remote_base_dirs=["base"],
        year_whitelist=None,
        validate_zip=True,
        max_workers=1,
    )

    assert result.n_jobs == 0
    assert result.n_files_total == 0
    assert result.n_written == 0
    assert result.n_skipped == 0
    assert result.n_failed == 0
    assert result.out_paths == []
    assert result.failed_paths == []
    assert instances and instances[0].closed is True


def test_ftp_download_threaded_path_runs_all_jobs(
    monkeypatch,
    tmp_path: Path,
    dummy_ftp_factory,
) -> None:
    mod = importlib.import_module("volatility_trading.etl.orats.ftp.download.run")

    base_to_years = {"base": ["2020", "2021"]}
    monkeypatch.setattr(
        mod,
        "FTP",
        dummy_ftp_factory(base_to_years, track_login=False),
    )

    def _result_factory(base: str, year_name: str):
        return mod.YearDownloadResult(
            base=base,
            year_name=year_name,
            n_files_total=1,
            n_written=1,
            n_skipped=0,
            n_failed=0,
            out_paths=[tmp_path / f"{base}-{year_name}.zip"],
            failed_paths=[],
        )

    monkeypatch.setattr(
        mod,
        "download_one_year",
        _make_download_one_year(mod, result_factory=_result_factory),
    )

    result = mod.download(
        user="u",
        password="p",
        raw_root=tmp_path,
        host="h",
        remote_base_dirs=["base"],
        year_whitelist=None,
        validate_zip=False,
        max_workers=2,
    )

    assert result.n_jobs == 2
    assert result.n_files_total == 2
    assert result.n_written == 2
    assert result.n_failed == 0
    assert len(result.out_paths) == 2

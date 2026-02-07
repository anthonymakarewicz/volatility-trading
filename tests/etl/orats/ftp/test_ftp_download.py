from __future__ import annotations

import importlib
from pathlib import Path


def test_ftp_download_filters_year_whitelist_and_aggregates(monkeypatch, tmp_path: Path) -> None:
    mod = importlib.import_module("volatility_trading.etl.orats.ftp.download.run")

    base_to_years = {"base": ["2019", "2020", "notes"]}

    class _DummyFTP:
        def __init__(self, host: str):
            self.host = host
            self.cwd_path = "/"
            self.closed = False

        def login(self, user: str, password: str) -> None:
            self.user = user
            self.password = password

        def cwd(self, path: str) -> None:
            if path == "/":
                self.cwd_path = "/"
            else:
                self.cwd_path = path

        def nlst(self):
            return list(base_to_years.get(self.cwd_path, []))

        def quit(self) -> None:
            self.closed = True

    monkeypatch.setattr(mod, "FTP", _DummyFTP)

    calls: list[tuple[str, str, bool]] = []

    def _download_one_year(*, base: str, year_name: str, validate_zip: bool, **kwargs):
        calls.append((base, year_name, validate_zip))
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

    monkeypatch.setattr(mod, "download_one_year", _download_one_year)

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


def test_ftp_download_no_jobs_returns_empty(monkeypatch, tmp_path: Path) -> None:
    mod = importlib.import_module("volatility_trading.etl.orats.ftp.download.run")

    base_to_years = {"base": ["README", "notes"]}
    instances: list[object] = []

    class _DummyFTP:
        def __init__(self, host: str):
            self.host = host
            self.cwd_path = "/"
            self.closed = False
            instances.append(self)

        def login(self, user: str, password: str) -> None:
            pass

        def cwd(self, path: str) -> None:
            if path == "/":
                self.cwd_path = "/"
            else:
                self.cwd_path = path

        def nlst(self):
            return list(base_to_years.get(self.cwd_path, []))

        def quit(self) -> None:
            self.closed = True

    monkeypatch.setattr(mod, "FTP", _DummyFTP)
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
    assert instances and getattr(instances[0], "closed") is True


def test_ftp_download_threaded_path_runs_all_jobs(monkeypatch, tmp_path: Path) -> None:
    mod = importlib.import_module("volatility_trading.etl.orats.ftp.download.run")

    base_to_years = {"base": ["2020", "2021"]}

    class _DummyFTP:
        def __init__(self, host: str):
            self.host = host
            self.cwd_path = "/"

        def login(self, user: str, password: str) -> None:
            pass

        def cwd(self, path: str) -> None:
            if path == "/":
                self.cwd_path = "/"
            else:
                self.cwd_path = path

        def nlst(self):
            return list(base_to_years.get(self.cwd_path, []))

        def quit(self) -> None:
            pass

    monkeypatch.setattr(mod, "FTP", _DummyFTP)

    def _download_one_year(*, base: str, year_name: str, **kwargs):
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

    monkeypatch.setattr(mod, "download_one_year", _download_one_year)

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
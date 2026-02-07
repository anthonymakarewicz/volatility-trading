from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest


@pytest.fixture()
def ftp_roots(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "raw", tmp_path / "out"


@pytest.fixture
def write_zip_files():
    def _write(year_dir: Path, names: Iterable[str]) -> None:
        for name in names:
            (year_dir / name).write_text("", encoding="utf-8")

    return _write


@pytest.fixture
def write_parquet_stub():
    def _write(self, file, *args, **kwargs) -> None:
        p = Path(file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")

    return _write


@pytest.fixture
def dummy_ftp_factory():
    def _factory(
        base_to_years: dict[str, list[str]],
        *,
        instances: list[object] | None = None,
        track_login: bool = True,
    ):
        class _DummyFTP:
            def __init__(self, host: str):
                self.host = host
                self.cwd_path = "/"
                self.closed = False
                self._base_to_years = base_to_years
                if instances is not None:
                    instances.append(self)

            def login(self, user: str, password: str) -> None:
                if track_login:
                    self.user = user
                    self.password = password

            def cwd(self, path: str) -> None:
                self.cwd_path = "/" if path == "/" else path

            def nlst(self):
                return list(self._base_to_years.get(self.cwd_path, []))

            def quit(self) -> None:
                self.closed = True

        return _DummyFTP

    return _factory

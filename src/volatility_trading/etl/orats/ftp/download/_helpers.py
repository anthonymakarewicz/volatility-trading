from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass
from ftplib import FTP
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class YearDownloadResult:
    base: str
    year_name: str
    n_files_total: int
    n_written: int
    n_skipped: int
    n_failed: int
    out_paths: list[Path]
    failed_paths: list[Path]


def _is_valid_zip(path: Path) -> bool:
    """Return True if `path` is a readable ZIP archive, else False."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            zf.namelist()
        return True
    except Exception:
        return False


def _download_atomic(ftp: FTP, remote_name: str, local_path: Path) -> None:
    """Download `remote_name` to `local_path` via temp file + atomic rename."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=str(local_path.parent),
            prefix=local_path.name + ".",
            suffix=".tmp",
        ) as f:
            tmp_path = f.name

            # Download directly into the temp file
            ftp.retrbinary("RETR " + remote_name, f.write)
            f.flush()
            os.fsync(f.fileno())

        if tmp_path is None:
            raise RuntimeError("Failed to allocate a temporary file")

        os.replace(tmp_path, local_path)

        # Best-effort directory fsync (POSIX)
        try:
            dir_fd = os.open(str(local_path.parent), os.O_DIRECTORY)
        except OSError:
            dir_fd = None

        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    except Exception:
        # Best-effort cleanup temp file
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def _ensure_file(
    ftp: FTP,
    remote_name: str,
    local_path: Path,
    *,
    validate_zip: bool = True,
) -> bool:
    """Ensure `local_path` is a complete, valid copy of the remote ZIP.

    Returns True if the file was downloaded (written), False if it was skipped.
    """
    remote_size: int | None = None

    if local_path.exists():
        try:
            remote_size = ftp.size(remote_name)
        except Exception as e:
            raise RuntimeError(f"Cannot get remote size for {remote_name}") from e

        local_size = local_path.stat().st_size
        if local_size == remote_size:
            if validate_zip and not _is_valid_zip(local_path):
                logger.warning(
                    "%s has correct size but invalid ZIP, re-downloading",
                    remote_name,
                )
                local_path.unlink(missing_ok=True)
            else:
                logger.info("%s already valid", remote_name)
                return False
        else:
            logger.warning(
                "%s partial or mismatched (%d != %d), re-downloading",
                remote_name,
                local_size,
                remote_size,
            )
            local_path.unlink(missing_ok=True)

    # Download (first-time or re-download)
    if remote_size is not None:
        mb = remote_size / (1024**2)
        logger.info("Downloading %s (%.2f MB)", remote_name, mb)
    else:
        logger.info("Downloading %s", remote_name)

    _download_atomic(ftp, remote_name, local_path)

    if validate_zip and not _is_valid_zip(local_path):
        # Treat as failure: remove corrupt file and raise
        local_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is not a valid ZIP: {local_path}")

    return True


def download_one_year(
    *,
    host: str,
    user: str,
    password: str,
    base: str,
    year_name: str,
    raw_root: Path,
    validate_zip: bool,
) -> YearDownloadResult:
    """Download all ZIP files for a given `(base, year_name)` into `raw_root`.

    Opens its own FTP connection so it can be executed safely in a thread.
    """
    logger.info("Starting FTP download base=%s year=%s", base, year_name)

    ftp = FTP(host)
    ftp.login(user, password)

    out_paths: list[Path] = []
    failed_paths: list[Path] = []
    n_files_total = 0
    n_written = 0
    n_skipped = 0
    n_failed = 0

    try:
        ftp.cwd("/")
        ftp.cwd(base)
        ftp.cwd(year_name)
        files = sorted(ftp.nlst())

        local_year_dir = raw_root / base / year_name
        local_year_dir.mkdir(parents=True, exist_ok=True)

        for remote_name in files:
            if not remote_name.lower().endswith(".zip"):
                continue

            n_files_total += 1
            local_path = local_year_dir / remote_name

            try:
                written = _ensure_file(
                    ftp,
                    remote_name,
                    local_path,
                    validate_zip=validate_zip,
                )
                if written:
                    n_written += 1
                    out_paths.append(local_path)
                else:
                    n_skipped += 1

            except Exception:
                n_failed += 1
                failed_paths.append(local_path)
                logger.error(
                    "Failed FTP download base=%s year=%s remote=%s path=%s",
                    base,
                    year_name,
                    remote_name,
                    local_path,
                    exc_info=True,
                )
                continue

    finally:
        try:
            ftp.quit()
        except Exception:
            pass

        logger.info(
            "Finished FTP download base=%s year=%s "
            "files=%d written=%d skipped=%d failed=%d",
            base,
            year_name,
            n_files_total,
            n_written,
            n_skipped,
            n_failed,
        )

    return YearDownloadResult(
        base=base,
        year_name=year_name,
        n_files_total=n_files_total,
        n_written=n_written,
        n_skipped=n_skipped,
        n_failed=n_failed,
        out_paths=out_paths,
        failed_paths=failed_paths,
    )
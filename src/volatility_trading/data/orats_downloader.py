from __future__ import annotations

import zipfile
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable
from ftplib import FTP
from pathlib import Path


def is_valid_zip(path: Path) -> bool:
    """
    Quick check whether a local file is a valid ZIP archive.

    Parameters
    ----------
    path : Path
        Path to the local .zip file.

    Returns
    -------
    bool
        True if the file can be opened as a ZIP and its namelist read, False otherwise.
    """
    try:
        with zipfile.ZipFile(path, "r") as zf:
            zf.namelist()
        return True
    except Exception:
        return False


def ensure_file(
    ftp: FTP,
    remote_name: str,
    local_path: Path,
    *,
    validate_zip: bool = True,
    verbose: bool = True,
) -> None:
    """
    Ensure that `local_path` is a correct, complete copy of `remote_name` on the FTP server.

    Logic:
    - If the local file exists:
        - Fetch the remote size.
        - If sizes match, optionally validate as ZIP; if valid, skip re-download.
        - If sizes differ or ZIP is invalid, delete and re-download.
    - If the local file does not exist:
        - Download it without querying the remote size (no MB log).

    Parameters
    ----------
    ftp : ftplib.FTP
        Active FTP connection positioned in the directory containing `remote_name`.
    remote_name : str
        File name on the FTP server (relative to current working directory).
    local_path : Path
        Local path to save the file.
    validate_zip : bool, optional
        If True, also verify that the local file is a valid ZIP archive, by default True.
    verbose : bool, optional
        If True, print progress messages, by default True.
    """
    remote_size: int | None = None

    if local_path.exists():
        # We need the remote size to decide whether the existing file is good
        try:
            remote_size = ftp.size(remote_name)
        except Exception:
            if verbose:
                print(f"    [skip] cannot get size for {remote_name}")
            return

        local_size = local_path.stat().st_size
        if local_size == remote_size:
            if validate_zip and not is_valid_zip(local_path):
                if verbose:
                    print(
                        f"    [warn] {remote_name} has correct size but invalid ZIP, "
                        "re-downloading"
                    )
                local_path.unlink(missing_ok=True)
            else:
                if verbose:
                    print(f"    [ok]   {remote_name} already valid")
                return
        else:
            if verbose:
                print(
                    f"    [warn] {remote_name} partial or mismatched "
                    f"({local_size} < {remote_size}), re-downloading"
                )
            local_path.unlink(missing_ok=True)

    # At this point: file did not exist, or we decided to re-download it.
    if verbose:
        if remote_size is not None:
            mb = remote_size / (1024**2)
            print(f"    [get]  {remote_name} ({mb:.2f} MB)")
        else:
            # No size fetched (e.g. first-time download) → no MB info
            print(f"    [get]  {remote_name}")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        ftp.retrbinary("RETR " + remote_name, f.write)


def _download_one_year(
    *,
    host: str,
    user: str,
    password: str,
    base: str,
    year_name: str,
    raw_root: Path,
    validate_zip: bool,
    verbose: bool,
) -> None:
    """
    Download all ZIP files for a given (base, year) pair into raw_root/base/year.

    This function opens its own FTP connection, so it can be safely run in a thread.
    """
    if verbose:
        print(f"[worker] Starting download for {base}/{year_name}")

    ftp = FTP(host)
    ftp.login(user, password)

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

            local_path = local_year_dir / remote_name
            ensure_file(
                ftp,
                remote_name,
                local_path,
                validate_zip=validate_zip,
                verbose=verbose,
            )

    finally:
        ftp.quit()
        if verbose:
            print(f"[worker] Finished {base}/{year_name}")


def download_orats_raw(
    *,
    host: str,
    user: str,
    password: str,
    remote_base_dirs: Iterable[str],
    raw_root: str | Path,
    year_whitelist: Iterable[int] | Iterable[str] | None = None,
    validate_zip: bool = True,
    verbose: bool = True,
    max_workers: int = 1,
) -> None:
    """
    Download ORATS raw ZIP files from HostedFTP into a local directory,
    in a restartable and optionally threaded way.

    Remote layout (example)
    -----------------------
    smvstrikes_2007_2012/
        2007/
            ...
        2012/
            ...
    smvstrikes/
        2013/
            ...
        2025/
            ...

    Local layout (mirrored)
    -----------------------
    <raw_root>/
        smvstrikes_2007_2012/
            2007/*.zip
            ...
        smvstrikes/
            2013/*.zip
            ...

    Parameters
    ----------
    host : str
        FTP host (e.g. "orats.hostedftp.com" or "de1.hostedftp.com").
    user : str
        FTP username.
    password : str
        FTP password.
    remote_base_dirs : Iterable[str]
        Top-level directories on the FTP server that contain year subdirectories,
        e.g. ["smvstrikes_2007_2012", "smvstrikes"].
    raw_root : str | Path
        Local root directory where raw ORATS ZIPs will be stored.
    year_whitelist : Iterable[int] | Iterable[str] | None, optional
        If given, only years in this collection will be downloaded.
        Values can be ints or strings (e.g. {2013, 2014} or {"2013", "2014"}).
    validate_zip : bool, optional
        If True, validate local files as ZIPs and re-download corrupt ones, by default True.
    verbose : bool, optional
        If True, print progress messages, by default True.
    max_workers : int, optional
        Number of worker threads to use. If set to 1, download is done sequentially
        (no threading). Values > 1 use a ThreadPoolExecutor, by default 1.
    """
    raw_root = Path(raw_root)

    # Normalize whitelist to a set of strings like {"2013", "2014"}
    if year_whitelist is not None:
        year_whitelist_str = {str(y) for y in year_whitelist}
    else:
        year_whitelist_str = None

    if verbose:
        print(f"Connecting to FTP host {host} to discover jobs...")

    ftp = FTP(host)
    ftp.login(user, password)

    jobs: list[tuple[str, str]] = []

    try:
        for base in remote_base_dirs:
            if verbose:
                print(f"\n=== Discovering years in base directory: {base} ===")

            ftp.cwd("/")
            ftp.cwd(base)

            years = sorted(ftp.nlst())
            if verbose:
                print("Years found:", years)

            for year_name in years:
                if not year_name.isdigit():
                    continue

                if year_whitelist_str is not None and year_name not in year_whitelist_str:
                    continue

                jobs.append((base, year_name))

    finally:
        ftp.quit()

    if not jobs:
        if verbose:
            print("No (base, year) jobs to download. Exiting.")
        return

    if verbose:
        print("\nJobs to download:", jobs)
        print(f"Using max_workers={max_workers}\n")

    # Sequential path (no threading)
    if max_workers <= 1:
        for base, year_name in jobs:
            _download_one_year(
                host=host,
                user=user,
                password=password,
                base=base,
                year_name=year_name,
                raw_root=raw_root,
                validate_zip=validate_zip,
                verbose=verbose,
            )
        if verbose:
            print("\nAll downloads completed (sequential).")
        return

    # Threaded path
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _download_one_year,
                host=host,
                user=user,
                password=password,
                base=base,
                year_name=year_name,
                raw_root=raw_root,
                validate_zip=validate_zip,
                verbose=verbose,
            )
            for (base, year_name) in jobs
        ]

        # This will raise the first exception encountered, if any
        for fut in futures:
            fut.result()

    if verbose:
        print("\nAll downloads completed (threaded).")
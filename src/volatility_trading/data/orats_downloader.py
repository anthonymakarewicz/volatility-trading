from __future__ import annotations

from ftplib import FTP
from pathlib import Path
from typing import Iterable
import zipfile


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
            # If this fails, the zip is corrupt
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

    - If the local file does not exist, it is downloaded.
    - If it exists but size or ZIP validity do not match, it is re-downloaded.
    - If it exists and passes checks, it is left untouched.

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
    # Ask remote for size
    try:
        remote_size = ftp.size(remote_name)
    except Exception:
        if verbose:
            print(f"    [skip] cannot get size for {remote_name}")
        return

    if local_path.exists():
        local_size = local_path.stat().st_size
        if local_size == remote_size:
            if validate_zip and not is_valid_zip(local_path):
                if verbose:
                    print(f"    [warn] {remote_name} has correct size but invalid ZIP, re-downloading")
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

    if verbose:
        mb = remote_size / (1024 ** 2)
        print(f"    [get]  {remote_name} ({mb:.2f} MB)")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        ftp.retrbinary("RETR " + remote_name, f.write)


def download_orats_raw(
    host: str,
    user: str,
    password: str,
    remote_base_dirs: Iterable[str],
    raw_root: str | Path,
    year_whitelist: Iterable[int] | Iterable[str] | None = None,
    *,
    validate_zip: bool = True,
    verbose: bool = True,
) -> None:
    """
    Download ORATS raw ZIP files from HostedFTP into a local directory,
    in a restartable and idempotent way.

    Remote layout (example)
    -----------------------
    smvstrikes_2007_2012/
        2007/
            ORATS_SMV_Strikes_20070103.zip
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
    """
    raw_root = Path(raw_root)

    # Normalize whitelist to a set of strings like {"2013", "2014"}
    if year_whitelist is not None:
        year_whitelist_str = {str(y) for y in year_whitelist}
    else:
        year_whitelist_str = None

    if verbose:
        print(f"Connecting to FTP host {host}...")

    ftp = FTP(host)
    ftp.login(user, password)

    if verbose:
        print("Connected.\n")

    try:
        for base in remote_base_dirs:
            if verbose:
                print(f"=== Base directory: {base} ===")

            # Ensure we start from the root for each base dir
            ftp.cwd("/")
            ftp.cwd(base)

            years = sorted(ftp.nlst())
            if verbose:
                print("Years found:", years)

            for year_name in years:
                # Skip non-year entries
                if not year_name.isdigit():
                    continue

                if year_whitelist_str is not None and year_name not in year_whitelist_str:
                    continue

                if verbose:
                    print(f"\n--- Year {base}/{year_name} ---")

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

                # Go back up to the base directory
                ftp.cwd("..")

            if verbose:
                print("")  # blank line between base dirs

    finally:
        if verbose:
            print("Closing FTP connection...")
        ftp.quit()
        if verbose:
            print("Done.")
"""Build cleaned OptionsDX panels from raw `.7z` archives."""

from __future__ import annotations

import re
import shutil
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import py7zr


def _normalize_column_name(name: str) -> str:
    """Normalize vendor column names to a stable snake-case-ish format."""
    return name.strip().replace("[", "").replace("]", "").lower()


def _to_float_expr(col: str) -> pl.Expr:
    """Robust numeric parse for space-padded vendor text fields."""
    return (
        pl.col(col)
        .cast(pl.String)
        .str.strip_chars()
        .str.replace_all(",", "")
        .cast(pl.Float64, strict=False)
        .alias(col)
    )


def extract_7z_and_load(
    archive_path: str | Path, low_memory: bool = True
) -> pl.DataFrame:
    """Extract one `.7z` archive and load non-empty `.txt` payloads with Polars."""
    archive_path = Path(archive_path)

    extract_root_dir = archive_path.parent / "_tmp_extract"
    extract_dir = extract_root_dir / archive_path.stem

    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(path=extract_dir)

    all_frames: list[pl.DataFrame] = []
    for file in extract_dir.glob("*.txt"):
        if file.stat().st_size == 0:
            print(f"Skipping empty file: {file}")
            continue
        print(f"Reading {file}")
        all_frames.append(pl.read_csv(file, low_memory=low_memory))

    shutil.rmtree(extract_dir, ignore_errors=True)

    if not all_frames:
        raise ValueError(f"No non-empty .txt files found in archive: {archive_path}")

    return pl.concat(all_frames, how="diagonal_relaxed")


def load_options(
    root: str | Path,
    start_year: int = 2012,
    end_year: int = 2022,
    low_memory: bool = True,
) -> pl.DataFrame:
    """Load raw OptionsDX archives for a year window as one Polars DataFrame."""
    raw_root_dir = Path(root)
    frames: list[pl.DataFrame] = []

    for year_dir in sorted(raw_root_dir.glob("*")):
        name = year_dir.name
        if not name.isdigit():
            continue
        year = int(name)
        if not (start_year <= year <= end_year):
            continue

        print(f"Processing year: {year}")
        for archive in sorted(year_dir.glob("*.7z")):
            print(f"  -> Extracting {archive.name}")
            frames.append(extract_7z_and_load(archive, low_memory=low_memory))

    if not frames:
        raise FileNotFoundError(
            f"No .7z archives found between years {start_year} and {end_year} in {root}"
        )

    df = pl.concat(frames, how="diagonal_relaxed")
    rename_map = {col: _normalize_column_name(col) for col in df.columns}
    df = df.rename(rename_map)

    if "quote_date" in df.columns:
        df = df.rename({"quote_date": "date"})
    if "expire_date" in df.columns:
        df = df.rename({"expire_date": "expiry"})

    return df


def reshape_options_wide_to_long(df: pl.DataFrame) -> pl.DataFrame:
    """Convert wide call/put columns (`c_*`, `p_*`) into a long options table."""
    call_cols = [col for col in df.columns if col.startswith("c_")]
    put_cols = [col for col in df.columns if col.startswith("p_")]
    shared_cols = [col for col in df.columns if col not in call_cols + put_cols]

    if not call_cols or not put_cols:
        raise ValueError(
            "Expected both call (c_*) and put (p_*) columns in wide OptionsDX input."
        )

    call_rows = (
        df.select(shared_cols + call_cols)
        .rename({col: col[2:] for col in call_cols})
        .with_columns(pl.lit("C").alias("option_type"))
    )
    put_rows = (
        df.select(shared_cols + put_cols)
        .rename({col: col[2:] for col in put_cols})
        .with_columns(pl.lit("P").alias("option_type"))
    )

    long_df = pl.concat([call_rows, put_rows], how="diagonal_relaxed")
    sort_cols = [c for c in ("date", "strike", "option_type") if c in long_df.columns]
    if sort_cols:
        long_df = long_df.sort(sort_cols)
    return long_df


def reshape_options_long_to_wide(df: pl.DataFrame) -> pl.DataFrame:
    """Pivot long options rows into wide call/put columns (`c_*`, `p_*`)."""
    if "option_type" not in df.columns:
        raise ValueError("Expected column 'option_type' not found.")

    index_candidates = [
        "date",
        "underlying_last",
        "expiry",
        "dte",
        "strike",
        "strike_distance",
        "strike_distance_pct",
    ]
    index_cols = [col for col in index_candidates if col in df.columns]
    value_cols = [
        col for col in df.columns if col not in set(index_cols + ["option_type"])
    ]
    if not value_cols:
        raise ValueError("No value columns available to pivot to wide format.")

    wide = df.pivot(
        values=value_cols,
        index=index_cols,
        on="option_type",
        aggregate_function="first",
    )

    rename_map: dict[str, str] = {}
    for col in wide.columns:
        match = re.fullmatch(r"(.+)_([CP])", col)
        if match:
            base, opt = match.groups()
            rename_map[col] = f"{opt.lower()}_{base.lower()}"
            continue
        if col in {"C", "P"} and len(value_cols) == 1:
            rename_map[col] = f"{col.lower()}_{value_cols[0].lower()}"
    if rename_map:
        wide = wide.rename(rename_map)

    sort_cols = [c for c in ("date", "strike") if c in wide.columns]
    if sort_cols:
        wide = wide.sort(sort_cols)
    return wide


def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """Clean wide OptionsDX data and return long-format options rows."""
    df = reshape_options_wide_to_long(df)

    cols_to_drop = [
        "quote_unixtime",
        "quote_readtime",
        "quote_time_hours",
        "expire_unix",
    ]
    existing_drop_cols = [col for col in cols_to_drop if col in df.columns]
    if existing_drop_cols:
        df = df.drop(existing_drop_cols)

    if "size" in df.columns:
        df = df.with_columns(
            [
                pl.col("size")
                .cast(pl.String)
                .str.strip_chars()
                .str.extract(r"(\d+)\s*[xX]\s*(\d+)", group_index=1)
                .cast(pl.Float64, strict=False)
                .alias("size_bid"),
                pl.col("size")
                .cast(pl.String)
                .str.strip_chars()
                .str.extract(r"(\d+)\s*[xX]\s*(\d+)", group_index=2)
                .cast(pl.Float64, strict=False)
                .alias("size_ask"),
            ]
        ).drop("size")

    dt_cols = ["date", "expiry"]
    dt_exprs: list[pl.Expr] = []
    for col in dt_cols:
        if col in df.columns:
            dt_exprs.append(
                pl.col(col)
                .cast(pl.String)
                .str.strip_chars()
                .str.to_datetime(strict=False)
                .alias(col)
            )
    if dt_exprs:
        df = df.with_columns(dt_exprs)

    numeric_cols = [c for c in df.columns if c not in set(dt_cols + ["option_type"])]
    if numeric_cols:
        df = df.with_columns([_to_float_expr(col) for col in numeric_cols])

    if "volume" in df.columns:
        df = df.with_columns(pl.col("volume").fill_null(0.0))

    if "ask" in df.columns and "bid" in df.columns:
        df = df.filter(pl.col("ask") >= pl.col("bid"))
    for col in ("bid", "ask", "last"):
        if col in df.columns:
            df = df.filter(pl.col(col) >= 0)

    if "dte" in df.columns:
        df = df.filter(pl.col("dte") >= 0).with_columns(pl.col("dte").round(0))

    sort_cols = [c for c in ("date", "strike") if c in df.columns]
    if sort_cols:
        df = df.sort(sort_cols)

    return df


def remove_illiquid_options(
    df: pl.DataFrame,
    volume_min: int = 1,
    rel_spread_max: float = 0.25,
    moneyness_band: tuple[float, float] = (0.8, 1.2),
) -> pl.DataFrame:
    """Filter out illiquid options via spread/volume/moneyness constraints."""
    required = ("bid", "ask", "volume", "strike", "underlying_last")
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for liquidity filter: {missing}")

    mid = 0.5 * (pl.col("bid") + pl.col("ask"))
    moneyness = pl.col("strike") / pl.col("underlying_last")
    return df.filter(
        (pl.col("bid") > 0)
        & (pl.col("ask") > 0)
        & (((pl.col("ask") - pl.col("bid")) / mid) <= rel_spread_max)
        & (pl.col("volume") >= volume_min)
        & (moneyness >= moneyness_band[0])
        & (moneyness <= moneyness_band[1])
    )


def extract_eom_options(df: pl.DataFrame) -> pl.DataFrame:
    """Return rows where expiry equals month-end of trade date."""
    if "date" not in df.columns or "expiry" not in df.columns:
        raise ValueError("Expected 'date' and 'expiry' columns.")

    return (
        df.with_columns(pl.col("date").dt.month_end().alias("eom_expiry"))
        .filter(pl.col("eom_expiry") == pl.col("expiry"))
        .drop("eom_expiry")
    )


def prepare_option_panel(
    *,
    raw_root: str | Path,
    output_path: str | Path,
    start_year: int = 2012,
    end_year: int = 2022,
    low_memory: bool = False,
    do_clean: bool = True,
    reshape: str | None = "wide",
    verbose: bool = True,
    return_df: bool = False,
) -> Path | pl.DataFrame:
    """Build one OptionsDX panel from raw archives and write parquet output."""
    raw_root = Path(raw_root)
    output_path = Path(output_path)

    if verbose:
        print(
            f"Loading options data from {raw_root} for years {start_year} to {end_year}..."
        )

    df = load_options(
        raw_root, start_year=start_year, end_year=end_year, low_memory=low_memory
    )

    if verbose:
        print(f"Loaded data shape: {df.shape}")

    if do_clean:
        if verbose:
            print("Cleaning data...")
        df = clean_data(df)
        if verbose:
            print(f"Data shape after cleaning: {df.shape}")

    if reshape == "wide":
        if verbose:
            print("Reshaping to wide format...")
        df = reshape_options_long_to_wide(df)
    elif reshape == "long":
        if verbose:
            print("Keeping long format...")
    elif reshape is None:
        if verbose:
            print("Skipping reshaping...")
    else:
        raise ValueError(
            f"Invalid reshape option '{reshape}'. Expected 'wide', 'long', or None."
        )

    if verbose:
        print(f"Writing Parquet to {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    if verbose:
        print("Finished writing panel.")

    if return_df:
        return df
    return output_path


def prepare_optionsdx_panels(
    *,
    raw_root: str | Path,
    proc_root: str | Path,
    tickers: Sequence[str] | None = None,
    start_year: int = 2012,
    end_year: int = 2022,
    low_memory: bool = False,
    do_clean: bool = True,
    reshape: str | None = "wide",
    verbose: bool = True,
    overwrite: bool = True,
    panel_name: str | None = None,
) -> dict[str, Path]:
    """Build one processed OptionsDX panel per ticker."""
    raw_root = Path(raw_root)
    proc_root = Path(proc_root)

    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw OptionsDX root not found: {raw_root}")
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year.")

    if tickers is None:
        discovered = sorted(path.name for path in raw_root.iterdir() if path.is_dir())
        if not discovered:
            raise FileNotFoundError(
                f"No ticker directories found under raw_root: {raw_root}"
            )
        selected_tickers = discovered
    else:
        selected_tickers = [str(t).strip() for t in tickers if str(t).strip()]
        if not selected_tickers:
            raise ValueError("tickers must not be empty when provided.")

    output_name = panel_name or f"optionsdx_panel_{start_year}_{end_year}.parquet"
    outputs: dict[str, Path] = {}

    for ticker in selected_tickers:
        ticker_raw_root = raw_root / ticker
        if not ticker_raw_root.is_dir():
            raise FileNotFoundError(
                f"Ticker raw directory not found: {ticker_raw_root}"
            )

        ticker_proc_root = proc_root / ticker
        output_path = ticker_proc_root / output_name
        if output_path.exists() and not overwrite:
            outputs[ticker] = output_path
            continue

        panel_result = prepare_option_panel(
            raw_root=ticker_raw_root,
            output_path=output_path,
            start_year=start_year,
            end_year=end_year,
            low_memory=low_memory,
            do_clean=do_clean,
            reshape=reshape,
            verbose=verbose,
            return_df=False,
        )
        if isinstance(panel_result, pl.DataFrame):
            raise RuntimeError(
                "prepare_option_panel returned a DataFrame unexpectedly. "
                "This path requires return_df=False."
            )
        outputs[ticker] = panel_result

    return outputs

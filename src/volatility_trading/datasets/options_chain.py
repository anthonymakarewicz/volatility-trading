"""
Small, opinionated I/O layer for the processed options chain dataset.

Conventions
----------
- scan_* returns a Polars LazyFrame
- read_* returns a Polars DataFrame
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from volatility_trading.config.paths import PROC_ORATS_OPTIONS_CHAIN


_DEFAULT_VALUE_COLS: tuple[str, ...] = (
    # quote fields
    "bid_price",
    "ask_price",
    "mid_price",
    "model_price",
    "rel_spread",
    # liquidity
    "volume",
    "open_interest",
    # greeks
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    # misc vendor fields (keep if present)
    "mid_iv",
)


def options_chain_path(proc_root: Path | str, ticker: str) -> Path:
    """
    Return the processed options chain parquet path for a ticker.
    """
    root = Path(proc_root)
    t = str(ticker).strip().upper()
    if not t:
        raise ValueError("ticker must be non-empty")
    return root / f"underlying={t}" / "part-0000.parquet"


def scan_options_chain(
    ticker: str,
    *,
    proc_root: Path | str = PROC_ORATS_OPTIONS_CHAIN,
    columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
    """Scan the processed **WIDE** ORATS options chain for one ticker (lazy).

    Parameters
    ----------
    ticker:
        Underlying symbol (e.g. "SPX", "SPY").
    proc_root:
        Root directory of the processed options chain dataset.
    columns:
        Optional column subset to project during the scan.

    Returns
    -------
    pl.LazyFrame
        LazyFrame pointing to `underlying=<TICKER>/part-0000.parquet`.

    Raises
    ------
    FileNotFoundError
        If the processed parquet does not exist.
    """
    root = Path(proc_root)
    path = options_chain_path(root, ticker)

    if not path.exists():
        raise FileNotFoundError(f"Processed options chain not found: {path}")

    lf = pl.scan_parquet(path)
    if columns is not None:
        lf = lf.select(list(columns))
    return lf


def read_options_chain(
    ticker: str,
    *,
    proc_root: Path | str = PROC_ORATS_OPTIONS_CHAIN,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Read the processed ORATS options chain for one ticker (WIDE, eager).

    Parameters
    ----------
    ticker:
        Underlying symbol (e.g. "SPX", "SPY").
    proc_root:
        Root directory of the processed dataset.
    columns:
        Optional list of columns to load.

    Returns
    -------
    pl.DataFrame
        WIDE options chain (call_* / put_* columns).
    """
    return (scan_options_chain(
        ticker, 
        proc_root=proc_root, 
        columns=columns).collect()
    )


def options_chain_wide_to_long(
    wide: pl.LazyFrame | pl.DataFrame
) -> pl.LazyFrame:
    """Convert a processed options chain from WIDE to LONG format.

    WIDE input uses `call_*` / `put_*` prefixed columns.
    LONG output standardizes names (prefix removed) and adds `option_type`
    in {"C", "P"}.

    Parameters
    ----------
    wide:
        WIDE options chain (LazyFrame or DataFrame).

    Returns
    -------
    pl.LazyFrame
        LONG options chain with `option_type`.
    """
    lf = wide.lazy() if isinstance(wide, pl.DataFrame) else wide

    schema = lf.collect_schema()
    cols = list(schema.names())

    call_cols = [c for c in cols if c.startswith("call_")]
    put_cols = [c for c in cols if c.startswith("put_")]
    base_cols = [c for c in cols if not c.startswith(("call_", "put_"))]

    if not call_cols and not put_cols:
        raise ValueError(
            "options_chain_wide_to_long expected at least one "
            "'call_*' or 'put_*' column, but found none."
        )

    calls = (
        lf.select(base_cols + call_cols)
        .rename({c: c.removeprefix("call_") for c in call_cols})
        .with_columns(pl.lit("C").alias("option_type"))
    )

    puts = (
        lf.select(base_cols + put_cols)
        .rename({c: c.removeprefix("put_") for c in put_cols})
        .with_columns(pl.lit("P").alias("option_type"))
    )

    long = pl.concat([calls, puts], how="vertical")
    return long.with_columns(pl.col("option_type").cast(pl.Categorical))


def options_chain_long_to_wide(
    long: pl.LazyFrame | pl.DataFrame,
    *,
    opt_col: str = "option_type",
    call_label: str = "C",
    put_label: str = "P",
    base_cols: Sequence[str] | None = None,
    value_cols: Sequence[str] | None = None,
    how: str = "inner",
) -> pl.LazyFrame:
    """Convert a LONG options chain into a WIDE chain with call_/put_ prefixes.
    
    This is the inverse helper of `options_chain_wide_to_long()`.

    Parameters
    ----------
    long:
        Long-format options chain (DataFrame or LazyFrame).
    opt_col:
        Option type column name (default: "option_type").
    call_label, put_label:
        Labels used inside `opt_col` to identify calls/puts.
    base_cols:
        Columns to join on (shared identifiers). If None, inferred as
        "all columns except opt_col and value_cols".
    value_cols:
        Per-option columns to widen. If None, uses a reasonable default set
        intersected with available columns.
    how:
        Join type between calls and puts ("inner" default is strict pairing).

    Returns
    -------
    pl.LazyFrame
        Wide-format chain with call_* and put_* columns.

    Notes
    -----
    - `how="inner"` keeps only rows where both call and put exist (best for parity).
    - `how="outer"` keeps rows even if one side is missing.
    """
    lf = long.lazy() if isinstance(long, pl.DataFrame) else long
    schema = lf.collect_schema()
    cols = schema.names()

    if opt_col not in cols:
        raise ValueError(f"options_chain_long_to_wide expected '{opt_col}' in columns.")

    # Infer value columns
    if value_cols is None:
        value_cols_eff = [c for c in _DEFAULT_VALUE_COLS if c in cols]
    else:
        value_cols_eff = [c for c in value_cols if c in cols]

    # Infer base cols
    if base_cols is None:
        base_cols_eff = [c for c in cols if c not in {opt_col, *value_cols_eff}]
    else:
        base_cols_eff = [c for c in base_cols if c in cols]

    if not base_cols_eff:
        raise ValueError("options_chain_long_to_wide could not infer base_cols (join keys).")

    def _widen_side(label: str, prefix: str) -> pl.LazyFrame:
        # keep base + values only
        sub = lf.filter(pl.col(opt_col) == label).select(base_cols_eff + value_cols_eff)
        # prefix value cols
        rename_map = {c: f"{prefix}{c}" for c in value_cols_eff}
        return sub.rename(rename_map)

    calls = _widen_side(call_label, "call_")
    puts = _widen_side(put_label, "put_")

    # Join on base cols so calls+puts end on the same row
    wide = calls.join(puts, on=base_cols_eff, how=how)
    return wide
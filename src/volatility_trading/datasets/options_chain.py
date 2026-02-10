"""I/O and reshape helpers for the processed ORATS options-chain panel.

Provides utilities to:
- resolve ticker-level parquet paths
- scan/read the processed wide chain
- convert between wide (`call_*` / `put_*`) and long (`option_type`) layouts
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import polars as pl

from volatility_trading.config.paths import PROC_ORATS_OPTIONS_CHAIN

JoinStrategy = Literal["inner", "left", "right", "full", "semi", "anti", "cross"]

DEFAULT_BASE_COLS = [
    "ticker",
    "trade_date",
    "expiry_date",
    "strike",
    "dte",
    "yte",
    "spot_price",
    "underlying_price",
    "risk_free_rate",
    "dividend_yield",
    "smoothed_iv",
]


def options_chain_path(proc_root: Path | str, ticker: str) -> Path:
    """Build the processed options-chain parquet path for one ticker.

    Args:
        proc_root: Root directory of the processed dataset.
        ticker: Underlying symbol (for example `SPX` or `AAPL`).

    Returns:
        Filesystem path to `underlying=<TICKER>/part-0000.parquet`.

    Raises:
        ValueError: If `ticker` is empty after normalization.
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
    """Scan the processed wide options-chain parquet lazily.

    Args:
        ticker: Underlying symbol to load.
        proc_root: Root directory of the processed options-chain dataset.
        columns: Optional projection list to select while scanning.

    Returns:
        Lazy frame backed by `underlying=<TICKER>/part-0000.parquet`.

    Raises:
        FileNotFoundError: If the ticker parquet file does not exist.
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
    """Read the processed wide options-chain parquet eagerly.

    Args:
        ticker: Underlying symbol to load.
        proc_root: Root directory of the processed options-chain dataset.
        columns: Optional projection list to load.

    Returns:
        Wide options-chain DataFrame with `call_*` and `put_*` columns.
    """
    return scan_options_chain(ticker, proc_root=proc_root, columns=columns).collect()


def options_chain_wide_to_long(wide: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """Convert a processed options chain from wide to long layout.

    Wide input uses `call_*` and `put_*` prefixed columns. Long output removes
    those prefixes and adds `option_type` in `{"C", "P"}`.

    Args:
        wide: Wide options-chain frame (lazy or eager).

    Returns:
        Long options-chain lazy frame with standardized per-leg columns.

    Raises:
        ValueError: If no `call_*` or `put_*` columns are present.
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
    how: JoinStrategy = "inner",
) -> pl.LazyFrame:
    """Convert a long options chain into a wide call/put representation.

    This is the inverse helper of `options_chain_wide_to_long`.

    Args:
        long: Long-format options chain (lazy or eager).
        opt_col: Column containing option-type labels.
        call_label: Label in `opt_col` representing calls.
        put_label: Label in `opt_col` representing puts.
        base_cols: Join keys used to align call and put rows.
        how: Join strategy between call and put subsets.

    Returns:
        Wide-format lazy frame with `call_*` and `put_*` value columns.

    Raises:
        ValueError: If `opt_col` is missing or no usable base keys are found.
    """
    lf = long.lazy() if isinstance(long, pl.DataFrame) else long
    schema = lf.collect_schema()
    cols = list(schema.names())

    if opt_col not in cols:
        raise ValueError(
            f"options_chain_long_to_wide expected '{opt_col}' in columns, "
            f"but found: {cols}"
        )

    # 1) Base cols: fixed default (ignore missing)
    if base_cols is None:
        base_cols_eff = [c for c in DEFAULT_BASE_COLS if c in cols]
    else:
        base_cols_eff = [c for c in base_cols if c in cols]

    if not base_cols_eff:
        raise ValueError(
            "options_chain_long_to_wide could not build base_cols (join keys). "
            "None of the provided/default base columns exist in the input."
        )

    # 2) Value cols = everything else except option_type and base cols
    excluded = {opt_col, *base_cols_eff}
    value_cols_eff = [c for c in cols if c not in excluded]

    if not value_cols_eff:
        # Still valid: you'll just join base keys (but no per-leg values).
        value_cols_eff = []

    def _widen_side(label: str, prefix: str) -> pl.LazyFrame:
        sub = lf.filter(pl.col(opt_col) == label).select(base_cols_eff + value_cols_eff)

        if value_cols_eff:
            rename_map = {c: f"{prefix}{c}" for c in value_cols_eff}
            sub = sub.rename(rename_map)

        return sub

    calls = _widen_side(call_label, "call_")
    puts = _widen_side(put_label, "put_")

    # 3) Join calls+puts on base keys
    wide = calls.join(puts, on=base_cols_eff, how=how)
    return wide

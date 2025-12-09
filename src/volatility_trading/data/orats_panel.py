from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence, Iterable

import polars as pl

from volatility_trading.config.schemas import (
    ORATS_VENDOR_TO_PROCESSED,
    CORE_ORATS_WIDE_COLUMNS,
    REQUIRED_ORATS_WIDE_COLUMNS,
)


def build_orats_panel_for_ticker(
    inter_root: Path | str,
    proc_root: Path | str,
    ticker: str,
    years: Iterable[int] | Iterable[str] | None = None,
    *,
    dte_min: int = 7,
    dte_max: int = 60,
    moneyness_min: float = 0.5,
    moneyness_max: float = 1.5,
    columns: Sequence[str] | None = None,
    verbose: bool = True,
) -> Path:
    """
    Build a cleaned, WIDE ORATS panel for a single ticker.

    Raw (intermediate) layout
    -------------------------
    inter_root/
        underlying=<TICKER>/year=<YYYY>/part-0000.parquet

    Processed output
    ----------------
    proc_root/
        orats_panel_<TICKER>.parquet

    The processed panel:
    - parses dates (trade_date, expiry_date)
    - computes DTE and K/S moneyness
    - normalises vendor names (stkPx -> spot_price, cBidPx -> call_bid, ...)
    - computes call/put mid prices, spreads, relative spreads
    - applies basic sanity filters
    - trims extreme DTE and moneyness
    - drops completely dead contracts (no quotes, no theo value)
    - exposes call Greeks as call_delta, call_gamma, ...
    - adds put Greeks via European put–call parity (put_delta, put_gamma, ...)

    Parameters
    ----------
    inter_root:
        Root of intermediate ORATS-by-ticker parquet structure.
    proc_root:
        Root for processed ORATS panels.
    ticker:
        Underlying symbol (e.g. "SPX").
    years:
        Optional subset of years to include (ints or strings).
    dte_min, dte_max:
        DTE band to keep in the processed panel.
    moneyness_min, moneyness_max:
        Coarse K/S moneyness band to keep.
    columns:
        Optional explicit list of columns for the output schema.
        If None, uses CORE_ORATS_WIDE_COLUMNS.
        REQUIRED_ORATS_WIDE_COLUMNS must always be included.
    verbose:
        If True, print basic progress messages.

    Returns
    -------
    Path
        Path to the written Parquet file.
    """
    inter_root = Path(inter_root)
    proc_root = Path(proc_root)

    pattern = inter_root / f"underlying={ticker}" / "year=*/part-0000.parquet"

    if verbose:
        print(f"\n=== Building ORATS WIDE panel for {ticker} ===")
        print(f"Reading from: {pattern}")

    # 0) Lazy scan
    scan = pl.scan_parquet(str(pattern))

    # 1) Normalise vendor names -> processed names
    lf = scan.rename(ORATS_VENDOR_TO_PROCESSED)

    # 2) Parse raw date strings (now using processed names)
    lf = lf.with_columns(
        trade_date = pl.col("trade_date").str.strptime(
            pl.Date, format="%m/%d/%Y", strict=False
        ),
        expiry_date = pl.col("expiry_date").str.strptime(
            pl.Date, format="%m/%d/%Y", strict=False
        ),
    )

    # 3) Optional year filter
    if years is not None:
        years_set = {int(y) for y in years}
        lf = lf.filter(pl.col("trade_date").dt.year().is_in(years_set))

    # 4) Derived features: DTE, moneyness, mids, spreads
    lf = lf.with_columns(
        dte = (pl.col("expiry_date") - pl.col("trade_date")).dt.total_days(),
        moneyness_ks = pl.col("strike") / pl.col("spot_price"),
        call_mid = (pl.col("call_bid") + pl.col("call_ask")) / 2.0,
        put_mid = (pl.col("put_bid") + pl.col("put_ask")) / 2.0,
        call_spread = pl.col("call_ask") - pl.col("call_bid"),
        put_spread  = pl.col("put_ask") - pl.col("put_bid"),
    )

    # relative spreads
    lf = lf.with_columns(
        call_rel_spread = (
            pl.when((pl.col("call_mid") > 0) & (pl.col("call_spread") >= 0))
            .then(pl.col("call_spread") / pl.col("call_mid"))
            .otherwise(None)
        ),
        put_rel_spread = (
            pl.when((pl.col("put_mid") > 0) & (pl.col("put_spread") >= 0))
            .then(pl.col("put_spread") / pl.col("put_mid"))
            .otherwise(None)
        ),
    )
    
    # 5) Filters: DTE band, sanity checks, moneyness trim, dead contracts
    lf = lf.filter(
        pl.col("dte").is_between(dte_min, dte_max),
        pl.col("spot_price") > 0,
        pl.col("strike") > 0,
        pl.col("call_ask") >= pl.col("call_bid"),
        pl.col("put_ask") >= pl.col("put_bid"),
        pl.col("moneyness_ks").is_between(moneyness_min, moneyness_max),
        ~(
            (pl.col("call_bid") == 0)
            & (pl.col("call_ask") == 0)
            & (pl.col("call_theo") == 0)
            & (pl.col("put_bid") == 0)
            & (pl.col("put_ask") == 0)
            & (pl.col("put_theo") == 0)
        ),
    )

    # 6) Put Greeks via European-style put–call parity 
    # (BS world with cont. div yield)
    disc_q = (-pl.col("dividend_yield") * pl.col("yte")).exp()
    disc_r = (-pl.col("risk_free_rate") * pl.col("yte")).exp()

    lf = lf.with_columns(
        put_delta = pl.col("call_delta") - disc_q,
        put_gamma = pl.col("call_gamma"),
        put_vega = pl.col("call_vega"),
        put_rho = pl.col("call_rho") - pl.col("yte") * pl.col("strike") * disc_r,
        put_theta = (
            pl.col("call_theta")
            + pl.col("dividend_yield") * pl.col("spot_price") * disc_q
            - pl.col("risk_free_rate") * pl.col("strike") * disc_r
        ),
    )

    # 7) Final selection + materialisation
    if columns is None:
        cols = CORE_ORATS_WIDE_COLUMNS
    else:
        cols = list(columns)
        missing = REQUIRED_ORATS_WIDE_COLUMNS - set(cols)
        if missing:
            raise ValueError(
                "build_orats_panel_for_ticker: missing required columns: "
                f"{sorted(missing)}"
            )

    df = lf.select(cols).collect()

    proc_root.mkdir(parents=True, exist_ok=True)
    out_path = proc_root / f"orats_panel_{ticker}.parquet"

    if verbose:
        print(
            f"Writing cleaned WIDE panel to: {out_path} "
            f"(rows={df.height}, cols={len(df.columns)})"
        )

    df.write_parquet(out_path)
    return out_path
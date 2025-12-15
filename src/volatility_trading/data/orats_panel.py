from __future__ import annotations

from collections.abc import Sequence, Iterable
from pathlib import Path

import polars as pl

from volatility_trading.config.constants import CALENDAR_DAYS_PER_YEAR
from volatility_trading.config.instruments import PREFERRED_OPRA_ROOT
from volatility_trading.config.schemas import (
    ORATS_VENDOR_TO_PROCESSED,
    CORE_ORATS_WIDE_COLUMNS,
)


def _scan_orats_intermediate_for_ticker(
    inter_root: Path | str,
    ticker: str,
    years: Iterable[int] | Iterable[str] | None = None,
    *,
    verbose: bool = True,
) -> pl.LazyFrame:
    """
    Build a lazy scan over intermediate ORATS files for a single ticker.

    Expected intermediate layout
    ----------------------------
    inter_root/
        underlying=<TICKER>/year=<YYYY>/part-0000.parquet
    """
    inter_root = Path(inter_root)
    root = inter_root / f"underlying={ticker}"

    if not root.exists():
        raise FileNotFoundError(
            f"No ORATS directory found for {ticker!r} at {root}"
        )

    if verbose:
        print(f"\n=== Building ORATS WIDE panel for {ticker} ===")
        print(f"Reading from: {root}")

    # optional year whitelist (as strings, e.g. {"2009", "2010"})
    if years is not None:
        year_whitelist = {str(y) for y in years}
    else:
        year_whitelist = None

    scans: list[pl.LazyFrame] = []

    for year_dir in sorted(root.glob("year=*")):
        if not year_dir.is_dir():
            continue

        # year_dir.name is like "year=2009"
        year_name = year_dir.name.split("=", 1)[-1]

        if year_whitelist is not None and year_name not in year_whitelist:
            continue

        part = year_dir / "part-0000.parquet"
        if not part.exists():
            if verbose:
                print(f"  [skip] missing {part}")
            continue

        if verbose:
            print(f"  [scan] {part}")
        scans.append(pl.scan_parquet(str(part)))

    if not scans:
        raise FileNotFoundError(
            f"No ORATS files found for {ticker!r} under {root}"
        )

    # allow schema evolution across years (e.g. cOpra/pOpra appear later)
    return pl.concat(scans, how="diagonal")


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
    - normalises vendor names (stkPx -> spot_price, cBidPx -> call_bid_price, ...)
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

    # --- 1) Scan intermediate per-year parquet files lazily ---
    scan = _scan_orats_intermediate_for_ticker(
        inter_root=inter_root,
        ticker=ticker,
        years=years,
        verbose=verbose,
    )

    # --- 2) Normalise vendor names -> processed names ---
    lf = scan.rename(ORATS_VENDOR_TO_PROCESSED)

    # --- 2b) Unify spot for index vs stock/ETF ---
    # For index options:
    #   spot_price      = cash index
    #   underlying_price = per-expiry forward
    # For stock/ETF options:
    #   spot_price may be null, and underlying_price is the tradable spot.
    lf = lf.with_columns(
        spot_price = pl.coalesce(
            [pl.col("spot_price"), pl.col("underlying_price")]
        )
    )

    # --- 3) Parse raw date strings into Date columns ---
    lf = lf.with_columns(
        trade_date = pl.col("trade_date").str.strptime(
            pl.Date, format="%m/%d/%Y", strict=False
        ),
        expiry_date = pl.col("expiry_date").str.strptime(
            pl.Date, format="%m/%d/%Y", strict=False
        ),
    )

    # --- 4) Derived features: DTE, moneyness, mids, spreads, rel spreads ---
    lf = lf.with_columns(
        dte = (pl.col("expiry_date") - pl.col("trade_date")).dt.total_days(),
        moneyness_ks = pl.col("strike") / pl.col("spot_price"),
        call_mid_price = (
            (pl.col("call_bid_price") + pl.col("call_ask_price")) / 2.0
        ),
        put_mid_price = (
            (pl.col("put_bid_price") + pl.col("put_ask_price")) / 2.0
        ),
        call_spread = pl.col("call_ask_price") - pl.col("call_bid_price"),
        put_spread  = pl.col("put_ask_price") - pl.col("put_bid_price"),
    )

    lf = lf.with_columns(
        call_rel_spread = (
            pl.when((pl.col("call_mid_price") > 0) & (pl.col("call_spread") >= 0))
            .then(pl.col("call_spread") / pl.col("call_mid_price"))
            .otherwise(None)
        ),
        put_rel_spread = (
            pl.when((pl.col("put_mid_price") > 0) & (pl.col("put_spread") >= 0))
            .then(pl.col("put_spread") / pl.col("put_mid_price"))
            .otherwise(None)
        ),
    )
    
    # --- 5) Filters: DTE band, sanity checks, moneyness trim, dead rows ---
    lf = lf.filter(
        pl.col("dte").is_between(dte_min, dte_max),
        pl.col("spot_price") > 0,
        pl.col("strike") > 0,
        pl.col("call_ask_price") >= pl.col("call_bid_price"),
        pl.col("put_ask_price") >= pl.col("put_bid_price"),
        pl.col("moneyness_ks").is_between(moneyness_min, moneyness_max),
        ~(
            (pl.col("call_bid_price") == 0)
            & (pl.col("call_ask_price") == 0)
            & (pl.col("call_model_price") == 0)
            & (pl.col("put_bid_price") == 0)
            & (pl.col("put_ask_price") == 0)
            & (pl.col("put_model_price") == 0)
        ),
    )

    # --- 6) Put greeks via European put–call parity ---
    disc_q = (-pl.col("dividend_yield") * pl.col("yte")).exp()
    disc_r = (-pl.col("risk_free_rate") * pl.col("yte")).exp()

    lf = lf.with_columns(
        put_delta = pl.col("call_delta") - disc_q,
        put_gamma = pl.col("call_gamma"),
        put_vega = pl.col("call_vega"),
        put_theta = (
            pl.col("call_theta") + (
                pl.col("dividend_yield") * pl.col("spot_price") * disc_q - 
                pl.col("risk_free_rate") * pl.col("strike") * disc_r
            ) / CALENDAR_DAYS_PER_YEAR
        ),
        put_rho = (
            pl.col("call_rho") - pl.col("yte") * pl.col("strike") * disc_r / 100
        )
    )

    # --- 7) Final column selection & materialisation ---
    if columns is None:
        cols = CORE_ORATS_WIDE_COLUMNS

    lf = lf.sort(["trade_date", "expiry_date", "strike"])
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
from __future__ import annotations

from collections.abc import Sequence, Iterable
from pathlib import Path

import polars as pl
import numpy as np

from volatility_trading.config.constants import CALENDAR_DAYS_PER_YEAR
from volatility_trading.config.instruments import PREFERRED_OPRA_ROOT
from volatility_trading.config.schemas import (
    ORATS_VENDOR_TO_PROCESSED,
    CORE_ORATS_WIDE_COLUMNS,
)

# TODO: Include the renaming of the ORATS cols + filter by extrem delta ([1%, 99%], DTE outside (1, 252))


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

    # --- 2a) Optional OPRA-root filtering (e.g. SPX vs SPXW) ---
    preferred_root = PREFERRED_OPRA_ROOT.get(ticker)
    if preferred_root is not None:
        lf = lf.filter(
            # For older years, call_opra is null / absent; keep those rows.
            pl.col("call_opra").is_null()
            | pl.col("call_opra").str.starts_with(preferred_root)
        )

    # --- 3) Unify spot for index vs stock/ETF ---
    """
    For index options:
        spot_price = cash index
        underlying_price = per-expiry parity implied forward
     For stock/ETF options:
       spot_price is null or 0.0, and underlying_price is the tradable spot.
    """
    lf = lf.with_columns(
        spot_price = pl.when(
            pl.col("spot_price").is_not_null() & (pl.col("spot_price") > 0)
        )
        .then(pl.col("spot_price"))
        .otherwise(pl.col("underlying_price"))
    )

    # --- 4) Parse raw date strings into Date columns ---
    lf = lf.with_columns(
        trade_date = pl.col("trade_date").str.strptime(
            pl.Date, format="%m/%d/%Y", strict=False
        ),
        expiry_date = pl.col("expiry_date").str.strptime(
            pl.Date, format="%m/%d/%Y", strict=False
        ),
    )

    # --- 5) Derived features: DTE, moneyness, mids, spreads, rel spreads ---
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
    
    # --- 6) Filters: DTE band, sanity checks, moneyness trim, dead rows ---
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

    # --- 7) Put greeks via European put–call parity ---
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

    # --- 8) Final column selection & materialisation ---
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


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[m], w[m]
    if x.size == 0:
        return np.nan
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w)
    cutoff = 0.5 * w.sum()
    return float(x[np.searchsorted(cw, cutoff)])


def _infer_q_term_structure(
    df: pl.DataFrame,
    *,
    group_cols=("trade_date", "dte"),
    strike_col="strike",
    spot_col="underlying_price",
    rate_col="risk_free_rate",
    yte_col="yte",
    call_price_col="call_model_price",
    put_price_col="put_model_price",
    call_delta_col="call_delta",
    # Optional realtive spread quality weight (set to None if not available)
    call_rel_spread_col: str | None = None,  # e.g. "call_rel_spread"
    put_rel_spread_col: str | None = None,   # e.g. "put_rel_spread"
    # Filters
    delta_lo=0.25,
    delta_hi=0.75,
    # Weights
    lambda_atm=40.0,   # set to 0.0 to disable ATM weighting
    eps=1e-12,
) -> pl.DataFrame:
    use_spreads = (call_rel_spread_col is not None) and (put_rel_spread_col is not None)

    # Base filters + compute qK and weights in Polars
    exprs = [
        pl.col(spot_col).alias("S"),
        pl.col(strike_col).alias("K"),
        pl.col(rate_col).alias("r"),
        pl.col(yte_col).alias("T"),
        pl.col(call_price_col).alias("C"),
        pl.col(put_price_col).alias("P"),
    ]

    lf = df.lazy().with_columns(exprs)

    # Delta band filter (calls only: typically in [0,1])
    lf = lf.filter(
        pl.col(call_delta_col).is_between(delta_lo, delta_hi),
        (pl.col("C") > 0) & (pl.col("P") > 0)
    )

    # e^{-qT} = ((C-P) + K e^{-rT}) / S
    lf = (
        lf
        .with_columns(
            (((pl.col("C") - pl.col("P")) + pl.col("K") * (-pl.col("r") * pl.col("T")).exp()) / pl.col("S"))
            .alias("e_m_qT"),
        )
        .filter(
            pl.col("e_m_qT").is_finite() & (pl.col("e_m_qT") > 0)
        )
    )

    # qK = -ln(e^{-qT}) / T
    lf = lf.with_columns([
        (-(pl.col("e_m_qT").log()) / pl.col("T")).alias("qK"),
    ]).filter(
        pl.col("qK").is_finite()
    )

    # Weights: (optional) “near-ATM” proxy via |ln(K/S)|  + (optional) spread weight if you have bid/ask
    lf = (
        lf
        .with_columns(
            (pl.col("K") / pl.col("S")).log().abs().alias("abs_logm")
        )
        .with_columns(
            (-pl.lit(lambda_atm) * pl.col("abs_logm")).exp().alias("w_atm")
        )
    )

    if use_spreads:
        # spread_rel may be NaN (your mid=0 case) => fallback to w_spread=1.0
        lf = (
            lf
            .with_columns([
                (pl.col(call_rel_spread_col) + pl.col(put_rel_spread_col)).alias("spread_rel")
            ])
            .with_columns([
                pl.when(pl.col("spread_rel").is_finite() & (pl.col("spread_rel") >= 0))
                  .then(1.0 / (pl.col("spread_rel") ** 2 + eps))
                  .otherwise(1.0)
                  .alias("w_spread")
            ])
            .with_columns([(pl.col("w_atm") * pl.col("w_spread")).alias("w")])
        )
    else:
        lf = lf.with_columns([pl.col("w_atm").alias("w")])

    core = lf.select([*group_cols, "T", "qK", "w"]).collect()
    
    out = (
        core
        .group_by(list(group_cols), maintain_order=True)
        .agg([
            pl.first("T").alias("T"),
            pl.len().alias("n_used"),
            pl.col("qK").alias("qK_list"),
            pl.col("w").alias("w_list"),
        ])
        .with_columns([
            pl.struct(["qK_list", "w_list"]).map_elements(
                lambda s: _weighted_median(np.array(s["qK_list"], float), np.array(s["w_list"], float)),
                return_dtype=pl.Float64,
            ).alias("q_implied"),

            pl.col("qK_list").map_elements(
                lambda xs: float(np.nanpercentile(np.array(xs, float), 25)) if len(xs) else np.nan,
                return_dtype=pl.Float64,
            ).alias("q_p25"),

            pl.col("qK_list").map_elements(
                lambda xs: float(np.nanpercentile(np.array(xs, float), 75)) if len(xs) else np.nan,
                return_dtype=pl.Float64,
            ).alias("q_p75"),
        ])
        .drop(["qK_list", "w_list"])
    )

    return out
from __future__ import annotations

import polars as pl
from pathlib import Path
from collections.abc import Sequence
from typing import Iterable

from volatility_trading.config.paths import INTER_ORATS_BY_TICKER, PROC_ORATS

# Default / core schema for the processed wide ORATS panel
CORE_ORATS_WIDE_COLUMNS = [
    "ticker",
    "trade_date",
    "expirDate",
    "dte",
    "yte",
    "stkPx",
    "strike",
    "moneyness_ks",
    "cVolu",
    "pVolu",
    "cBidPx",
    "cAskPx",
    "cMidPx",
    "pBidPx",
    "pAskPx",
    "pMidPx",
    "cSpread",
    "pSpread",
    "cRelSpread",
    "pRelSpread",
    "cBidIv",
    "cMidIv",
    "cAskIv",
    "pBidIv",
    "pMidIv",
    "pAskIv",
    "smoothSmvVol",
    "iRate",
    "divRate",
    "extVol",
    # call Greeks (renamed from ORATS raw greeks)
    "cDelta",
    "cGamma",
    "cTheta",
    "cVega",
    "cRho",
    # put Greeks (via parity)
    "pDelta",
    "pGamma",
    "pTheta",
    "pVega",
    "pRho",
]

# Minimal set of columns that must always be present
REQUIRED_ORATS_WIDE_COLUMNS = {
    "ticker",
    "trade_date",
    "expirDate",
    "dte",
    "stkPx",
    "strike",
}


def build_orats_panel_for_ticker(
    ticker: str,
    inter_root: Path | str = INTER_ORATS_BY_TICKER,
    proc_root: Path | str = PROC_ORATS,
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

    - parses dates
    - computes DTE and K/S moneyness
    - computes call/put mid prices and spreads
    - applies basic sanity filters
    - trims extreme DTE and moneyness
    - drops completely dead contracts
    - renames ORATS call Greeks to cDelta, cGamma, ...
    - adds put Greeks via European put-call parity (pDelta, pGamma, ...)

    Parameters
    ----------
    ticker:
        Underlying symbol (e.g. "SPX").
    inter_root:
        Root of intermediate ORATS-by-ticker parquet structure.
        Expected layout:
            inter_root/
                underlying=<TICKER>/year=<YYYY>/part-0000.parquet
    proc_root:
        Root for processed ORATS panels.
    years:
        Optional subset of years to include (ints or strings).
    dte_min, dte_max:
        DTE band to keep in the processed panel.
    moneyness_min, moneyness_max:
        Coarse K/S moneyness band to keep.
    columns:
        Optional explicit list of columns to keep in the output.
        If None, uses CORE_ORATS_WIDE_COLUMNS.
        REQUIRED_ORATS_WIDE_COLUMNS must always be included, otherwise a
        ValueError is raised.
    verbose:
        If True, print basic progress messages.

    Returns
    -------
    Path to the written Parquet file.
    """
    inter_root = Path(inter_root)
    proc_root = Path(proc_root)

    pattern = inter_root / f"underlying={ticker}" / "year=*/part-0000.parquet"

    if verbose:
        print(f"\n=== Building ORATS WIDE panel for {ticker} ===")
        print(f"Reading from: {pattern}")

    scan = pl.scan_parquet(str(pattern))

    # 1) Parse dates (and optionally filter years)
    lf = scan.with_columns(
        pl.col("trade_date")
        .str.strptime(pl.Date, fmt="%m/%d/%Y", strict=False)
        .alias("trade_date"),
        pl.col("expirDate")
        .str.strptime(pl.Date, fmt="%m/%d/%Y", strict=False)
        .alias("expirDate"),
    )

    if years is not None:
        years_set = {int(y) for y in years}
        lf = lf.filter(pl.col("trade_date").dt.year().is_in(years_set))

    # 2) Basic derived features: DTE, moneyness, mids, spreads, rel spreads
    lf = lf.with_columns(
        # days to expiry
        (pl.col("expirDate") - pl.col("trade_date"))
        .dt.days()
        .alias("dte"),

        # K/S moneyness
        (pl.col("strike") / pl.col("stkPx")).alias("moneyness_ks"),

        # mid prices
        ((pl.col("cBidPx") + pl.col("cAskPx")) / 2.0).alias("cMidPx"),
        ((pl.col("pBidPx") + pl.col("pAskPx")) / 2.0).alias("pMidPx"),

        # spreads
        (pl.col("cAskPx") - pl.col("cBidPx")).alias("cSpread"),
        (pl.col("pAskPx") - pl.col("pBidPx")).alias("pSpread"),

        # relative spreads (for later entry filters)
        (
            pl.when((pl.col("cMidPx") > 0) & (pl.col("cSpread") >= 0))
            .then(pl.col("cSpread") / pl.col("cMidPx"))
            .otherwise(None)
        ).alias("cRelSpread"),
        (
            pl.when((pl.col("pMidPx") > 0) & (pl.col("pSpread") >= 0))
            .then(pl.col("pSpread") / pl.col("pMidPx"))
            .otherwise(None)
        ).alias("pRelSpread"),
    )

    # 3) Filters: DTE band, sanity checks, moneyness trim, dead contracts
    lf = lf.filter(
        # DTE band
        pl.col("dte").is_between(dte_min, dte_max),

        # hard sanity checks
        pl.col("stkPx") > 0,
        pl.col("strike") > 0,
        pl.col("cAskPx") >= pl.col("cBidPx"),
        pl.col("pAskPx") >= pl.col("pBidPx"),

        # coarse moneyness trim
        pl.col("moneyness_ks").is_between(moneyness_min, moneyness_max),

        # drop contracts totally dead on BOTH sides
        ~(
            (pl.col("cBidPx") == 0)
            & (pl.col("cAskPx") == 0)
            & (pl.col("cValue") == 0)  # adjust to your actual column names
            & (pl.col("pBidPx") == 0)
            & (pl.col("pAskPx") == 0)
            & (pl.col("pValue") == 0)
        ),
    )

    # 4) Rename ORATS call Greeks to cDelta, cGamma, ...
    # (we keep the original columns in the lazy frame, but will not select them)
    lf = lf.with_columns(
        pl.col("delta").alias("cDelta"),
        pl.col("gamma").alias("cGamma"),
        pl.col("theta").alias("cTheta"),
        pl.col("vega").alias("cVega"),
        pl.col("rho").alias("cRho"),
    )

    # 5) Put Greeks via parity (inline, no helper)
    # European-style parity using BS world with continuous div yield
    disc_q = (-pl.col("divRate") * pl.col("yte")).exp()
    disc_r = (-pl.col("iRate") * pl.col("yte")).exp()

    lf = lf.with_columns(
        pDelta = pl.col("cDelta") - disc_q,
        pGamma = pl.col("cGamma"),
        pVega  = pl.col("cVega"),
        pRho   = pl.col("cRho") - pl.col("yte") * pl.col("strike") * disc_r,
        pTheta = (
            pl.col("cTheta")
            + pl.col("divRate") * pl.col("stkPx") * disc_q
            - pl.col("iRate") * pl.col("strike") * disc_r
        ),
    )

    # 6) Final selection + materialisation
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
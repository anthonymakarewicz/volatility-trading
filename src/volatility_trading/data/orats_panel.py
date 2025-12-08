from __future__ import annotations

import polars as pl
from pathlib import Path
from typing import Iterable

from volatility_trading.config.paths import INTER_ORATS_BY_TICKER, PROC_ORATS


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
    - adds call (ORATS) + put (parity) Greeks
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

    # 4) Put Greeks via parity (inline, no helper)
    disc_q = (-pl.col("divRate") * pl.col("yte")).exp()
    disc_r = (-pl.col("iRate") * pl.col("yte")).exp()

    lf = lf.with_columns(
        pDelta = pl.col("delta") - disc_q,
        pGamma = pl.col("gamma"),
        pVega  = pl.col("vega"),
        pRho   = pl.col("rho") - pl.col("yte") * pl.col("strike") * disc_r,
        pTheta = (
            pl.col("theta")
            + pl.col("divRate") * pl.col("stkPx") * disc_q
            - pl.col("iRate") * pl.col("strike") * disc_r
        ),
    )

    # 5) Final selection + materialisation
    df = lf.select(
        [
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
            # call Greeks (ORATS)
            pl.col("delta").alias("cDelta"),
            pl.col("gamma").alias("cGamma"),
            pl.col("theta").alias("cTheta"),
            pl.col("vega").alias("cVega"),
            pl.col("rho").alias("cRho"),
            # put Greeks (parity, already named)
            "pDelta",
            "pGamma",
            "pTheta",
            "pVega",
            "pRho",
        ]
    ).collect()

    proc_root.mkdir(parents=True, exist_ok=True)
    out_path = proc_root / f"orats_panel_{ticker}.parquet"

    if verbose:
        print(
            f"Writing cleaned WIDE panel to: {out_path} "
            f"(rows={df.height}, cols={len(df.columns)})"
        )

    df.write_parquet(out_path)
    return out_path
"""Put-greek derivation steps for processed options-chain panels."""

from __future__ import annotations

import polars as pl

from volatility_trading.config.constants import CALENDAR_DAYS_PER_YEAR


def add_put_greeks(*, lf: pl.LazyFrame) -> pl.LazyFrame:
    """Derive put Greeks via European put–call parity."""
    disc_q = (-pl.col("dividend_yield") * pl.col("yte")).exp()
    disc_r = (-pl.col("risk_free_rate") * pl.col("yte")).exp()

    return lf.with_columns(
        put_delta=pl.col("call_delta") - disc_q,
        put_gamma=pl.col("call_gamma"),
        put_vega=pl.col("call_vega"),
        put_theta=(
            pl.col("call_theta")
            + (
                pl.col("dividend_yield") * pl.col("spot_price") * disc_q
                - pl.col("risk_free_rate") * pl.col("strike") * disc_r
            )
            / CALENDAR_DAYS_PER_YEAR
        ),
        put_rho=(pl.col("call_rho") - pl.col("yte") * pl.col("strike") * disc_r / 100),
    )


def add_put_greeks_simple(*, lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add put Greeks using a minimal convention.

    This is useful when vendor Greeks are defined per-strike with a single
    greek set (ORATS philosophy). We keep gamma/vega/theta/rho identical
    between calls and puts and adjust delta by a constant shift.
    """
    return lf.with_columns(
        put_delta=pl.col("call_delta") - 1.0,
        put_gamma=pl.col("call_gamma"),
        put_vega=pl.col("call_vega"),
        put_theta=pl.col("call_theta"),
        put_rho=pl.col("call_rho"),
    )

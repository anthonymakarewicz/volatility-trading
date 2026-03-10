"""Feature engineering helpers for skew mispricing strategy inputs."""

from __future__ import annotations

import pandas as pd

from volatility_trading.backtesting.data_contracts import HedgeMarketData

SUPPORTED_ORATS_SUMMARY_TENORS: dict[int, str] = {
    10: "10d",
    20: "20d",
    30: "30d",
    60: "60d",
    90: "90d",
    182: "6m",
    365: "1y",
}


def resolve_orats_summary_tenor_suffix(target_dte: int) -> str:
    """Return the ORATS daily-features tenor suffix for one supported DTE."""
    try:
        return SUPPORTED_ORATS_SUMMARY_TENORS[int(target_dte)]
    except KeyError as exc:
        supported = ", ".join(str(dte) for dte in SUPPORTED_ORATS_SUMMARY_TENORS)
        raise ValueError(
            "skew_mispricing only supports target_dte values backed by ORATS "
            f"daily features: {supported}"
        ) from exc


def required_skew_feature_columns(target_dte: int) -> tuple[str, str]:
    """Return the required ORATS feature columns for one raw RR skew tenor."""
    tenor_suffix = resolve_orats_summary_tenor_suffix(target_dte)
    return (
        f"iv_dlt25_{tenor_suffix}",
        f"iv_dlt75_{tenor_suffix}",
    )


def build_skew_signal_input(
    options: pd.DataFrame,
    features: pd.DataFrame | None,
    hedge_market: HedgeMarketData | None,
    *,
    target_dte: int = 30,
) -> pd.Series:
    """Build raw 25-delta risk-reversal skew aligned to options trading dates.

    Args:
        options: Normalized options panel indexed by trade date.
        features: Daily features panel containing ORATS summary IV columns.
        hedge_market: Unused hedge data hook required by the strategy contract.
        target_dte: Supported ORATS tenor used for both the skew signal and
            risk-reversal structure defaults.

    Returns:
        Series indexed by options trading dates containing raw skew
        ``iv_dlt25_<tenor> - iv_dlt75_<tenor>``. Missing feature dates remain
        null; values are never forward-filled.

    Raises:
        ValueError: If ``features`` is missing, improperly indexed, or does not
            contain the required ORATS skew columns.
    """
    _ = hedge_market
    if features is None:
        raise ValueError(
            "skew_mispricing requires data.features with ORATS daily skew columns"
        )

    put_col, call_col = required_skew_feature_columns(target_dte)
    missing = [
        column for column in (put_col, call_col) if column not in features.columns
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            "skew_mispricing requires daily feature columns: " f"{missing_list}"
        )

    feature_frame = features.copy()
    feature_frame.index = _coerce_trade_date_index(feature_frame)
    raw_skew = pd.to_numeric(feature_frame[put_col], errors="coerce") - pd.to_numeric(
        feature_frame[call_col], errors="coerce"
    )
    raw_skew = raw_skew.groupby(level=0).last().sort_index()
    trading_dates = pd.DatetimeIndex(
        pd.to_datetime(options.index.unique())
    ).sort_values()
    tenor_suffix = resolve_orats_summary_tenor_suffix(target_dte)
    return raw_skew.reindex(trading_dates).rename(f"skew_rr_{tenor_suffix}")


def _coerce_trade_date_index(features: pd.DataFrame) -> pd.DatetimeIndex:
    """Return a datetime index keyed by trade date for one features frame."""
    if isinstance(features.index, pd.MultiIndex):
        if "trade_date" not in features.index.names:
            raise ValueError(
                "skew_mispricing requires data.features to be indexed by trade_date"
            )
        return pd.DatetimeIndex(
            pd.to_datetime(features.index.get_level_values("trade_date"))
        )
    return pd.DatetimeIndex(pd.to_datetime(features.index))

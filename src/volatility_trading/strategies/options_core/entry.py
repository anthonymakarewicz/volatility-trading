"""Generic entry-intent builders from structure specifications."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from volatility_trading.backtesting import BacktestConfig

from .adapters import option_type_to_chain_label
from .selectors import choose_expiry_by_target_dte, pick_quote_by_delta
from .types import EntryIntent, LegSelection, LegSpec, StructureSpec


def chain_for_date(options: pd.DataFrame, trade_date: pd.Timestamp) -> pd.DataFrame:
    """Return a date slice as DataFrame even when a single row is returned."""
    chain = options.loc[trade_date]
    if isinstance(chain, pd.Series):
        return chain.to_frame().T
    return chain


def normalize_signals_to_on(
    signals: pd.DataFrame | pd.Series,
    *,
    strategy_name: str = "Strategy",
) -> pd.DataFrame:
    """Normalize signals to a sorted DataFrame with a boolean `on` column."""
    if isinstance(signals, pd.Series):
        sig_df = signals.to_frame("on")
    else:
        sig_df = signals.copy()

    if "on" not in sig_df.columns:
        if "short" in sig_df.columns:
            sig_df["on"] = sig_df["short"].astype(bool)
        elif "long" in sig_df.columns:
            sig_df["on"] = sig_df["long"].astype(bool)
        elif sig_df.shape[1] == 1:
            sig_df["on"] = sig_df.iloc[:, 0].astype(bool)
        else:
            raise ValueError(
                f"{strategy_name} expects a boolean 'on' column in signals, "
                "or a Series, or a DF with 'short'/'long' or a single column."
            )
    return sig_df.sort_index()


def _entry_price_from_side(
    quote: pd.Series, *, side: int, cfg: BacktestConfig
) -> float:
    """Return executable entry price for one leg given target side."""
    if side == -1:
        return float(quote["bid_price"] - cfg.slip_bid)
    if side == 1:
        return float(quote["ask_price"] + cfg.slip_ask)
    raise ValueError("side must be -1 (short) or +1 (long)")


def _apply_leg_liquidity_filters(
    leg_quotes: pd.DataFrame, leg_spec: LegSpec
) -> pd.DataFrame:
    """Apply optional per-leg liquidity/spread filters from `LegSpec`."""
    filtered = leg_quotes

    if leg_spec.min_open_interest > 0 and "open_interest" in filtered.columns:
        filtered = filtered[filtered["open_interest"] >= leg_spec.min_open_interest]
    if leg_spec.min_volume > 0 and "volume" in filtered.columns:
        filtered = filtered[filtered["volume"] >= leg_spec.min_volume]
    if (
        leg_spec.max_relative_spread is not None
        and "bid_price" in filtered.columns
        and "ask_price" in filtered.columns
    ):
        mids = 0.5 * (filtered["bid_price"] + filtered["ask_price"])
        rel_spread = (filtered["ask_price"] - filtered["bid_price"]) / mids.where(
            mids > 0,
            np.nan,
        )
        filtered = filtered[rel_spread <= leg_spec.max_relative_spread]

    return filtered


def build_entry_intent_from_structure(
    *,
    entry_date: pd.Timestamp,
    options: pd.DataFrame,
    structure_spec: StructureSpec,
    cfg: BacktestConfig,
    side_resolver: Callable[[LegSpec], int],
    features: pd.DataFrame | None = None,
    fallback_iv_feature_col: str = "iv_atm",
    min_atm_quotes: int = 2,
) -> EntryIntent | None:
    """Build one `EntryIntent` from chain data and structure constraints.

    Selection policy:
    - choose expiry nearest `structure_spec.dte_target` within tolerance
    - per leg: filter by type/liquidity/spread, then pick by delta target
    - fill policy: `all_or_none` or `min_ratio`
    """
    chain = chain_for_date(options, entry_date)
    chosen_dte = choose_expiry_by_target_dte(
        chain=chain,
        target_dte=structure_spec.dte_target,
        max_dte_diff=structure_spec.dte_tolerance,
        min_atm_quotes=min_atm_quotes,
    )
    if chosen_dte is None:
        return None

    chain = chain[chain["dte"] == chosen_dte]
    expiry_date = pd.Timestamp(chain["expiry_date"].iloc[0])
    selected_legs: list[LegSelection] = []
    total_legs = len(structure_spec.legs)

    for leg_spec in structure_spec.legs:
        option_type_label = option_type_to_chain_label(leg_spec.option_type)
        leg_quotes = chain[chain["option_type"] == option_type_label]
        leg_quotes = _apply_leg_liquidity_filters(leg_quotes, leg_spec)
        leg_q = pick_quote_by_delta(
            leg_quotes,
            target_delta=leg_spec.delta_target,
            delta_tolerance=leg_spec.delta_tolerance,
        )
        if leg_q is None:
            continue
        side = int(side_resolver(leg_spec))
        selected_legs.append(
            LegSelection(
                spec=leg_spec,
                quote=leg_q,
                side=side,
                entry_price=_entry_price_from_side(leg_q, side=side, cfg=cfg),
            )
        )

    if not selected_legs:
        return None

    fill_ratio = len(selected_legs) / total_legs
    if structure_spec.fill_policy == "all_or_none" and len(selected_legs) != total_legs:
        return None
    if (
        structure_spec.fill_policy == "min_ratio"
        and fill_ratio < structure_spec.min_fill_ratio
    ):
        return None

    if "spot_price" in chain.columns and not chain.empty:
        spot_entry = float(chain["spot_price"].iloc[0])
    else:
        spot_entry = float("nan")

    if selected_legs and all("smoothed_iv" in leg.quote.index for leg in selected_legs):
        iv_entry = float(
            sum(float(leg.quote["smoothed_iv"]) for leg in selected_legs)
            / len(selected_legs)
        )
    elif (
        features is not None
        and entry_date in features.index
        and fallback_iv_feature_col in features.columns
    ):
        iv_entry = float(features.loc[entry_date, fallback_iv_feature_col])
    else:
        iv_entry = float("nan")

    return EntryIntent(
        entry_date=entry_date,
        expiry_date=expiry_date,
        chosen_dte=int(chosen_dte),
        legs=tuple(selected_legs),
        spot=spot_entry,
        volatility=iv_entry,
    )

"""Entry-intent builders that resolve a structure spec against chain data.

The module translates abstract structure templates into concrete, tradable legs:
- normalize strategy signals into directional entry events,
- select one expiry per expiry-group and best quotes per leg,
- apply fill-policy constraints,
- emit an ``EntryIntent`` consumed by sizing and lifecycle engines.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from volatility_trading.options import MarketState, PositionSide

from ..config import BacktestRunConfig
from .contracts.market import QuoteSnapshot
from .contracts.structures import (
    EntryIntent,
    LegSelection,
    LegSpec,
    StructureSpec,
)
from .selectors import select_best_expiry_for_leg_group


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
    """Normalize signals to a sorted DataFrame with `on` and `entry_direction`.

    `entry_direction` is normalized to {-1, 0, +1} where:
    - `-1`: short structure entry
    - `+1`: long structure entry
    - `0`: no entry
    """
    if isinstance(signals, pd.Series):
        sig_df = signals.to_frame(signals.name or "value")
    else:
        sig_df = signals.copy()

    has_long = "long" in sig_df.columns
    has_short = "short" in sig_df.columns
    has_entry_direction = "entry_direction" in sig_df.columns
    long_on = (
        sig_df["long"].astype(bool)
        if has_long
        else pd.Series(False, index=sig_df.index)
    )
    short_on = (
        sig_df["short"].astype(bool)
        if has_short
        else pd.Series(False, index=sig_df.index)
    )

    if has_entry_direction:
        direction = pd.to_numeric(sig_df["entry_direction"], errors="coerce").fillna(
            0.0
        )
        direction = np.sign(direction).astype(int)
    elif has_long and has_short:
        direction = pd.Series(0, index=sig_df.index, dtype=int)
        direction.loc[long_on & ~short_on] = 1
        direction.loc[short_on & ~long_on] = -1
    elif has_short:
        direction = pd.Series(0, index=sig_df.index, dtype=int)
        direction.loc[short_on] = -1
    elif has_long:
        direction = pd.Series(0, index=sig_df.index, dtype=int)
        direction.loc[long_on] = 1
    else:
        raise ValueError(
            f"{strategy_name} signals must include `entry_direction` "
            "or `long`/`short` columns."
        )

    if "on" in sig_df.columns:
        sig_df["on"] = sig_df["on"].astype(bool)
    elif has_long or has_short:
        sig_df["on"] = long_on | short_on
    else:
        sig_df["on"] = direction != 0

    # Defensive normalization: never allow direction != 0 when `on` is False.
    direction = direction.where(sig_df["on"], 0).astype(int)
    sig_df["entry_direction"] = direction

    if bool((sig_df["on"] & (sig_df["entry_direction"] == 0)).any()):
        raise ValueError(
            f"{strategy_name} produced `on=True` with `entry_direction=0`."
        )

    return sig_df.sort_index()


def _entry_price_from_side(
    quote: QuoteSnapshot, *, side: PositionSide, cfg: BacktestRunConfig
) -> float:
    """Return executable entry price for one leg given target side."""
    if side == PositionSide.SHORT:
        return float(quote.bid_price - cfg.execution.slip_bid)
    if side == PositionSide.LONG:
        return float(quote.ask_price + cfg.execution.slip_ask)
    raise ValueError("side must be PositionSide.SHORT or PositionSide.LONG")


def _legs_grouped_by_expiry(
    structure_spec: StructureSpec,
) -> dict[str, list[tuple[int, LegSpec]]]:
    """Group legs by expiry key while preserving original leg order."""
    grouped: dict[str, list[tuple[int, LegSpec]]] = {}
    for idx, leg_spec in enumerate(structure_spec.legs):
        grouped.setdefault(leg_spec.expiry_group, []).append((idx, leg_spec))
    return grouped


def _resolve_group_dte_constraints(
    *,
    group_name: str,
    group_legs: tuple[LegSpec, ...],
    structure_spec: StructureSpec,
) -> tuple[int, int]:
    """Resolve one expiry-group DTE target/tolerance from leg or structure values.

    Raises:
        ValueError: If legs in the same expiry group define conflicting DTE values.
    """
    explicit_targets = {
        leg.dte_target for leg in group_legs if leg.dte_target is not None
    }
    explicit_tolerances = {
        leg.dte_tolerance for leg in group_legs if leg.dte_tolerance is not None
    }

    if len(explicit_targets) > 1:
        raise ValueError(
            f"expiry_group={group_name!r} has conflicting dte_target values"
        )
    if len(explicit_tolerances) > 1:
        raise ValueError(
            f"expiry_group={group_name!r} has conflicting dte_tolerance values"
        )

    target_dte = (
        int(next(iter(explicit_targets)))
        if explicit_targets
        else int(structure_spec.dte_target)
    )
    dte_tolerance = (
        int(next(iter(explicit_tolerances)))
        if explicit_tolerances
        else int(structure_spec.dte_tolerance)
    )
    return target_dte, dte_tolerance


def build_entry_intent_from_structure(
    *,
    entry_date: pd.Timestamp,
    options: pd.DataFrame,
    structure_spec: StructureSpec,
    cfg: BacktestRunConfig,
    side_resolver: Callable[[LegSpec], int | PositionSide],
    features: pd.DataFrame | None = None,
    fallback_iv_feature_col: str = "iv_atm",
) -> EntryIntent | None:
    """Build one `EntryIntent` from chain data and structure constraints.

    Selection policy:
    - choose one expiry per `expiry_group`
    - enforce hard delta/liquidity filters per leg
    - score feasible legs by delta distance then liquidity
    - score feasible expiries by DTE distance plus weighted average leg score
    - apply structure-level fill policy (`all_or_none` or `min_ratio`)
    """
    chain = chain_for_date(options, entry_date)
    total_legs = len(structure_spec.legs)
    selected_by_index: dict[int, LegSelection] = {}
    group_results: dict[str, tuple[pd.Timestamp, int]] = {}

    grouped_legs = _legs_grouped_by_expiry(structure_spec)
    for group_name, indexed_group_legs in grouped_legs.items():
        group_legs = tuple(leg_spec for _, leg_spec in indexed_group_legs)
        target_dte, dte_tolerance = _resolve_group_dte_constraints(
            group_name=group_name,
            group_legs=group_legs,
            structure_spec=structure_spec,
        )
        selected_group = select_best_expiry_for_leg_group(
            chain=chain,
            group_legs=group_legs,
            target_dte=target_dte,
            dte_tolerance=dte_tolerance,
        )
        if selected_group is None:
            continue

        expiry_date_group, chosen_dte_group, quotes_for_group = selected_group
        group_results[group_name] = (expiry_date_group, chosen_dte_group)
        for (leg_idx, leg_spec), quote in zip(
            indexed_group_legs,
            quotes_for_group,
            strict=True,
        ):
            side_raw = side_resolver(leg_spec)
            try:
                side = (
                    side_raw
                    if isinstance(side_raw, PositionSide)
                    else PositionSide(int(side_raw))
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "side_resolver must return PositionSide.SHORT/-1 or PositionSide.LONG/+1"
                ) from exc
            selected_by_index[leg_idx] = LegSelection(
                spec=leg_spec,
                quote=quote,
                side=side,
                entry_price=_entry_price_from_side(quote, side=side, cfg=cfg),
            )

    selected_legs = tuple(
        selected_by_index[idx] for idx in sorted(selected_by_index.keys())
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

    primary_group = "main" if "main" in group_results else next(iter(group_results))
    expiry_date, chosen_dte = group_results[primary_group]

    if "spot_price" in chain.columns and not chain.empty:
        spot_entry = float(chain["spot_price"].iloc[0])
    else:
        spot_entry = float("nan")

    if selected_legs and all(leg.quote.market_iv is not None for leg in selected_legs):
        iv_entry = float(
            sum(float(leg.quote.market_iv) for leg in selected_legs)
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
        legs=selected_legs,
        entry_state=MarketState(spot=spot_entry, volatility=iv_entry),
    )

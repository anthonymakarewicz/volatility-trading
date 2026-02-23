"""Generic entry-intent builders from structure specifications."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from volatility_trading.backtesting import BacktestConfig

from .selectors import select_best_expiry_for_leg_group
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
    """Resolve group DTE settings from leg overrides or structure defaults."""
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
    cfg: BacktestConfig,
    side_resolver: Callable[[LegSpec], int],
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
            side = int(side_resolver(leg_spec))
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
        legs=selected_legs,
        spot=spot_entry,
        volatility=iv_entry,
    )

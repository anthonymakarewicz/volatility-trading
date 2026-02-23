"""Shared quote/expiry selection helpers for option structure builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .adapters import option_type_to_chain_label
from .types import LegSpec


def apply_leg_liquidity_filters(
    leg_quotes: pd.DataFrame,
    *,
    leg_spec: LegSpec,
) -> pd.DataFrame:
    """Apply hard liquidity filters for one leg selection."""
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
        mids = 0.5 * (
            filtered["bid_price"] + filtered["ask_price"]
        )  # TODO: Use rel_spread that we already have in the chain else compute it
        rel_spread = (filtered["ask_price"] - filtered["bid_price"]) / mids.where(
            mids > 0,
            np.nan,
        )
        filtered = filtered[rel_spread <= leg_spec.max_relative_spread]

    return filtered


def score_leg_candidates(
    leg_quotes: pd.DataFrame,
    *,
    leg_spec: LegSpec,
) -> pd.DataFrame:
    """Score feasible candidates for one leg using target-delta then liquidity."""
    if leg_quotes.empty or "delta" not in leg_quotes.columns:
        return pd.DataFrame()

    scored = leg_quotes.copy()
    scored["delta_distance"] = (
        scored["delta"] - leg_spec.delta_target
    ).abs() / leg_spec.delta_tolerance
    scored = scored[scored["delta_distance"] <= 1.0]
    if scored.empty:
        return scored

    scored = apply_leg_liquidity_filters(scored, leg_spec=leg_spec)
    if scored.empty:
        return scored

    if (
        "bid_price" in scored.columns and "ask_price" in scored.columns
    ):  # TODO: Computation here is repeated could have a small helper for that
        mids = 0.5 * (scored["bid_price"] + scored["ask_price"])
        rel_spread = (scored["ask_price"] - scored["bid_price"]) / mids.where(
            mids > 0,
            np.nan,
        )
    else:
        rel_spread = pd.Series(0.0, index=scored.index)

    rel_spread = rel_spread.replace([np.inf, -np.inf], np.nan)
    scored["rel_spread"] = rel_spread.fillna(np.inf)
    finite_spread = rel_spread.dropna()
    spread_scale = float(finite_spread.max()) if not finite_spread.empty else 0.0
    if spread_scale > 0:
        scored["spread_norm"] = (rel_spread / spread_scale).fillna(1.0)
    else:
        scored["spread_norm"] = 0.0

    if "open_interest" in scored.columns and float(scored["open_interest"].max()) > 0:
        oi_norm = np.log1p(scored["open_interest"]) / np.log1p(
            float(scored["open_interest"].max())
        )
    else:
        oi_norm = pd.Series(0.0, index=scored.index)
    if "volume" in scored.columns and float(scored["volume"].max()) > 0:
        vol_norm = np.log1p(scored["volume"]) / np.log1p(float(scored["volume"].max()))
    else:
        vol_norm = pd.Series(0.0, index=scored.index)

    scored["depth_norm"] = 0.5 * (oi_norm + vol_norm)
    scored["leg_score"] = (
        scored["delta_distance"]
        + 0.25 * scored["spread_norm"]
        + 0.10 * (1.0 - scored["depth_norm"])
    )
    return scored.sort_values(
        by=["leg_score", "delta_distance", "rel_spread"],
        ascending=[True, True, True],
    )


def select_best_quote_for_leg(
    chain_for_expiry: pd.DataFrame,
    *,
    leg_spec: LegSpec,
) -> tuple[pd.Series, float] | None:
    """Return best quote and score for one leg inside one expiry candidate."""
    option_type_label = option_type_to_chain_label(leg_spec.option_type)
    leg_quotes = chain_for_expiry[chain_for_expiry["option_type"] == option_type_label]
    scored = score_leg_candidates(leg_quotes, leg_spec=leg_spec)
    if scored.empty:
        return None

    score = float(scored.iloc[0]["leg_score"])
    best = scored.iloc[0].drop(
        labels=[
            c
            for c in (
                "delta_distance",
                "rel_spread",
                "spread_norm",
                "depth_norm",
                "leg_score",
            )
            if c in scored.columns
        ]
    )
    return best, score


def select_best_expiry_for_leg_group(
    *,
    chain: pd.DataFrame,
    group_legs: tuple[LegSpec, ...],
    target_dte: int,
    dte_tolerance: int,
    expiry_column: str = "expiry_date",
    dte_column: str = "dte",
) -> tuple[pd.Timestamp, int, tuple[pd.Series, ...]] | None:
    """Pick one expiry for a leg group, then best quotes for all group legs.

    Candidate expiries are constrained by DTE tolerance and must be feasible for
    all legs in the group. Feasible expiries are scored by DTE distance plus
    weighted average of per-leg quote scores.
    """
    if not group_legs:
        return None
    if expiry_column not in chain.columns or dte_column not in chain.columns:
        return None

    expiry_stats = (
        chain[[expiry_column, dte_column]]
        .dropna(subset=[expiry_column, dte_column])
        .assign(_expiry_ts=lambda df: pd.to_datetime(df[expiry_column]))
        .groupby("_expiry_ts", as_index=False)[dte_column]
        .median()
    )
    if expiry_stats.empty:
        return None

    candidate_expiries = expiry_stats[
        (expiry_stats[dte_column] - target_dte).abs() <= dte_tolerance
    ]
    if candidate_expiries.empty:
        return None

    best_payload: (
        tuple[
            float,
            float,
            pd.Timestamp,
            int,
            tuple[pd.Series, ...],
        ]
        | None
    ) = None

    chain_with_expiry_ts = chain.assign(
        _expiry_ts=pd.to_datetime(chain[expiry_column]),
    )
    for _, row in candidate_expiries.iterrows():
        expiry_ts = pd.Timestamp(row["_expiry_ts"])
        expiry_dte = int(row[dte_column])
        chain_for_expiry = chain_with_expiry_ts[
            chain_with_expiry_ts["_expiry_ts"] == expiry_ts
        ]

        quotes_for_group: list[pd.Series] = []
        leg_scores: list[float] = []
        weights: list[float] = []
        feasible = True
        for leg_spec in group_legs:
            best_leg = select_best_quote_for_leg(
                chain_for_expiry,
                leg_spec=leg_spec,
            )
            if best_leg is None:
                feasible = False
                break
            quote, score = best_leg
            quotes_for_group.append(quote)
            leg_scores.append(score)
            weights.append(float(abs(int(leg_spec.weight))))
        if not feasible:
            continue

        leg_score = (
            float(np.average(leg_scores, weights=weights))
            if weights
            else float(np.mean(leg_scores))
        )
        dte_distance_norm = abs(expiry_dte - target_dte) / max(1, dte_tolerance)
        total_score = dte_distance_norm + leg_score

        payload = (
            total_score,
            dte_distance_norm,
            expiry_ts,
            expiry_dte,
            tuple(quotes_for_group),
        )
        if best_payload is None or payload[:3] < best_payload[:3]:
            best_payload = payload

    if best_payload is None:
        return None

    _, _, expiry_ts, expiry_dte, quotes = best_payload
    return expiry_ts, expiry_dte, quotes

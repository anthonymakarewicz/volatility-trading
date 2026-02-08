from __future__ import annotations

import polars as pl

from .expr_helpers import _all_not_null, _apply_required_mask

# -----------------------------------------------------------------------------
# Theoretical Upper/Lower bounds checks
# -----------------------------------------------------------------------------


def _bounds_dyn_abs_tol(
    *,
    bid_col: str,
    ask_col: str,
    multiplier: float = 0.5,
    floor: float = 0.01,
) -> pl.Expr:
    """
    Dynamic absolute tolerance based on bid/ask spread.

    tol = max(floor, multiplier * (ask - bid))
    """
    spread = (pl.col(ask_col) - pl.col(bid_col)).abs()
    return pl.max_horizontal([pl.lit(floor), pl.lit(multiplier) * spread])


def flag_option_bounds_mid_eu_forward(
    df: pl.DataFrame,
    *,
    option_type: str,
    mid_col: str = "mid_price",
    bid_col: str = "bid_price",
    ask_col: str = "ask_price",
    forward_col: str = "underlying_price",  # F0
    strike_col: str = "strike",
    r_col: str = "risk_free_rate",
    yte_col: str = "yte",
    out_col: str = "price_bounds_mid_eu_violation",
    multiplier: float = 0.5,
    tol_floor: float = 0.01,
) -> pl.DataFrame:
    """
    European option theoretical bounds using the forward price F0.

    For calls:
        max(0, exp(-rT)*(F0 - K)) <= C_mid <= exp(-rT)*F0

    For puts:
        max(0, exp(-rT)*(K - F0)) <= P_mid <= exp(-rT)*K

    Flags violations using a dynamic absolute tolerance derived from bid/ask spread.
    """
    disc_r = (-pl.col(r_col) * pl.col(yte_col)).exp()

    is_call = pl.lit(str(option_type).upper() == "C")

    lower = (
        pl.when(is_call)
        .then(
            (disc_r * (pl.col(forward_col) - pl.col(strike_col))).clip(lower_bound=0.0)
        )
        .otherwise(
            (disc_r * (pl.col(strike_col) - pl.col(forward_col))).clip(lower_bound=0.0)
        )
    )

    upper = (
        pl.when(is_call)
        .then(disc_r * pl.col(forward_col))
        .otherwise(disc_r * pl.col(strike_col))
    )

    tol = _bounds_dyn_abs_tol(
        bid_col=bid_col,
        ask_col=ask_col,
        multiplier=multiplier,
        floor=tol_floor,
    )

    mid = pl.col(mid_col)
    violation = (mid < (lower - tol)) | (mid > (upper + tol))

    required = _all_not_null(
        [mid_col, bid_col, ask_col, forward_col, strike_col, r_col, yte_col]
    )

    return df.with_columns(
        _apply_required_mask(
            violation_expr=violation,
            required_expr=required,
            out_col=out_col,
        )
    )


def flag_option_bounds_mid_am_spot(
    df: pl.DataFrame,
    *,
    option_type: str,
    mid_col: str = "mid_price",
    bid_col: str = "bid_price",
    ask_col: str = "ask_price",
    spot_col: str = "spot_price",  # S0
    strike_col: str = "strike",
    out_col: str = "price_bounds_mid_am_spot_violation",
    multiplier: float = 0.5,
    tol_floor: float = 0.01,
) -> pl.DataFrame:
    """
    American option theoretical bounds (spot-based).

    Call bounds:
        max(0, S0 - K) <= C_mid <= S0

    Put bounds:
        max(0, K - S0) <= P_mid <= K

    Flags violations using a dynamic absolute tolerance derived from bid/ask spread.
    """
    is_call = pl.lit(str(option_type).upper() == "C")

    call_intrinsic = (pl.col(spot_col) - pl.col(strike_col)).clip(lower_bound=0.0)
    put_intrinsic = (pl.col(strike_col) - pl.col(spot_col)).clip(lower_bound=0.0)

    lower = pl.when(is_call).then(call_intrinsic).otherwise(put_intrinsic)
    upper = pl.when(is_call).then(pl.col(spot_col)).otherwise(pl.col(strike_col))

    tol = _bounds_dyn_abs_tol(
        bid_col=bid_col,
        ask_col=ask_col,
        multiplier=multiplier,
        floor=tol_floor,
    )

    mid = pl.col(mid_col)
    violation = (mid < (lower - tol)) | (mid > (upper + tol))

    required = _all_not_null([mid_col, bid_col, ask_col, spot_col, strike_col])

    return df.with_columns(
        _apply_required_mask(
            violation_expr=violation,
            required_expr=required,
            out_col=out_col,
        )
    )

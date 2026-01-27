# qc/specs_soft.py
from __future__ import annotations

from .dataset_checks import (
    check_missing_sessions_xnys,
    check_non_trading_dates_present_xnys,
    check_unique_rf_rate_per_day_expiry,
    check_spot_constant_per_trade_date,
    check_forward_constant_per_trade_date_expiry,
    check_spot_equals_underlying_per_trade_date_am,
)
from .row_checks import (
    flag_locked_market,
    flag_one_sided_quotes,
    flag_wide_spread,
    flag_zero_vol_pos_oi,
    flag_pos_vol_zero_oi,
    flag_delta_bounds,
    flag_theta_positive,
    flag_iv_high,
    flag_strike_monotonicity,
    flag_maturity_monotonicity,
    flag_option_bounds_mid_eu_forward,
    flag_option_bounds_mid_am_spot,
    flag_put_call_parity_mid_eu_forward,
    flag_put_call_parity_bounds_mid_am,
)
from .spec_types import SoftDatasetSpec, SoftRowSpec, SoftSpec


BASE_KEYS = [
    "trade_date",
    "expiry_date",
    "strike",
    "option_type",
    "dte",
    "delta",
    "yte",
    "underlying_price",
    "spot_price",
    "risk_free_rate",
    "dividend_yield",
]


def _get_base_soft_specs() -> list[SoftSpec]:
    return [
        # ---- Quote diagnostics ----
        SoftRowSpec(
            base_name="locked_market",
            flagger=flag_locked_market,
            violation_col="locked_market_violation",
            flagger_kwargs={},
            use_roi=True,
            by_option_type=True,
            sample_cols=BASE_KEYS + ["bid_price", "ask_price"],
        ),
        SoftRowSpec(
            base_name="one_sided_quotes",
            flagger=flag_one_sided_quotes,
            violation_col="one_sided_quote_violation",
            flagger_kwargs={},
            use_roi=True,
            by_option_type=True,
            sample_cols=BASE_KEYS + ["bid_price", "ask_price"],
        ),
        SoftRowSpec(
            base_name="wide_spread",
            flagger=flag_wide_spread,
            violation_col="wide_spread_violation",
            flagger_kwargs={"threshold": 1.0, "min_mid": 0.01},
            use_roi=True,
            by_option_type=True,
            sample_cols=BASE_KEYS + ["bid_price", "ask_price", "mid_price"],
        ),
        SoftRowSpec(
            base_name="very_wide_spread",
            flagger=flag_wide_spread,
            violation_col="wide_spread_violation",
            flagger_kwargs={"threshold": 2.0, "min_mid": 0.01},
            use_roi=True,
            by_option_type=True,
            sample_cols=BASE_KEYS + ["bid_price", "ask_price", "mid_price"],
        ),

        # ---- Volume / OI diagnostics ----
        SoftRowSpec(
            base_name="zero_vol_pos_oi",
            flagger=flag_zero_vol_pos_oi,
            thresholds={"mild": 0.05, "warn": 0.15, "fail": 0.30},
            violation_col="zero_vol_pos_oi_violation",
            flagger_kwargs={},
            use_roi=True,
            by_option_type=True,
            sample_cols=BASE_KEYS + ["volume", "open_interest"],
        ),
        SoftRowSpec(
            base_name="pos_vol_zero_oi",
            flagger=flag_pos_vol_zero_oi,
            thresholds={"mild": 0.01, "warn": 0.03, "fail": 0.05},
            violation_col="pos_vol_zero_oi_violation",
            flagger_kwargs={},
            use_roi=True,
            by_option_type=True,
            sample_cols=BASE_KEYS + ["volume", "open_interest"],
        ),

        # ---- Greeks sign diagnostics ----
        SoftRowSpec(
            base_name="delta_bounds_sane",
            flagger=flag_delta_bounds,
            thresholds={"mild": 1e-6, "warn": 1e-5, "fail": 1e-4},
            violation_col="delta_bounds_violation",
            flagger_kwargs={"eps": 1e-5},
            use_roi=True,
            by_option_type=True,
        ),
        SoftRowSpec(
            base_name="theta_positive",
            flagger=flag_theta_positive,
            thresholds={"mild": 1e-3, "warn": 0.005, "fail": 0.01},
            violation_col="theta_positive_violation",
            flagger_kwargs={"eps": 1e-8},
            use_roi=True,
            by_option_type=True,
            sample_cols=BASE_KEYS + ["theta"],
        ),

        # ---- IV diagnostics ----
        SoftRowSpec(
            base_name="high_iv",
            flagger=flag_iv_high,
            thresholds={"mild": 1e-3, "warn": 0.005, "fail": 0.01},
            violation_col="iv_too_high_violation",
            flagger_kwargs={"threshold": 1.0},
            use_roi=False,
            by_option_type=False,
            sample_cols=BASE_KEYS + ["smoothed_iv"],
        ),
        SoftRowSpec(
            base_name="very_high_iv",
            flagger=flag_iv_high,
            thresholds={"mild": 1e-6, "warn": 1e-5, "fail": 1e-4},
            violation_col="iv_too_high_violation",
            flagger_kwargs={"threshold": 2.0},
            use_roi=False,
            by_option_type=False,
            sample_cols=BASE_KEYS + ["smoothed_iv"],
        ),

        # ---- Arbitrage diagnostics ----
        SoftRowSpec(
            base_name="strike_monotonicity",
            flagger=flag_strike_monotonicity,
            thresholds={"mild": 0.01, "warn": 0.05, "fail": 0.10},
            violation_col="strike_monot_violation",
            flagger_kwargs={"price_col": "mid_price"},
            use_roi=True,
            by_option_type=True,
            sample_cols=BASE_KEYS + ["mid_price"],
        ),
        SoftRowSpec(
            base_name="maturity_monotonicity",
            flagger=flag_maturity_monotonicity,
            thresholds={"mild": 0.01, "warn": 0.05, "fail": 0.10},
            violation_col="maturity_monot_violation",
            flagger_kwargs={"price_col": "mid_price"},
            use_roi=True,
            by_option_type=True,
            sample_cols=BASE_KEYS + ["mid_price"],
        ),

        # ---- Dataset-level row_checks (GLOBAL only) ----
        SoftDatasetSpec(
            base_name="missing_sessions_xnys",
            checker=check_missing_sessions_xnys,
            thresholds={"mild": 0.002, "warn": 0.01, "fail": 0.03},
            use_roi=False,
        ),
        SoftDatasetSpec(
            base_name="non_trading_dates_present_xnys",
            checker=check_non_trading_dates_present_xnys,
            thresholds={"mild": 1e-5, "warn": 1e-3, "fail": 1e-2},
            use_roi=False,
        ),
        SoftDatasetSpec(
            base_name="unique_risk_free_rate_per_day_expiry",
            checker=check_unique_rf_rate_per_day_expiry,
            checker_kwargs={"tol_abs": 1e-4, "tol_rel": 0.0},
            thresholds={"mild": 0.001, "warn": 0.01, "fail": 0.05},
            use_roi=False,
        ),
        SoftDatasetSpec(
            base_name="spot_constant_per_trade_date",
            checker=check_spot_constant_per_trade_date,
            checker_kwargs={"tol_abs": 0.001, "tol_rel": 5e-4},
            thresholds={"mild": 0.001, "warn": 0.01, "fail": 0.05},
            use_roi=False,
        ),
    ]


def _get_exercise_soft_specs(exercise_style: str | None) -> list[SoftSpec]:
    if exercise_style == "EU":
        return [
            SoftDatasetSpec(
                base_name="forward_constant_per_trade_date_expiry",
                checker=check_forward_constant_per_trade_date_expiry,
                checker_kwargs={"tol_abs": 0.001,"tol_rel": 5e-4},
                thresholds={"mild": 0.001, "warn": 0.01, "fail": 0.05},
                use_roi=False,
            ),
            SoftRowSpec(
                base_name="price_bounds_mid_eu_forward",
                flagger=flag_option_bounds_mid_eu_forward,
                thresholds={"mild": 0.05, "warn": 0.10, "fail": 0.20},
                violation_col="price_bounds_mid_eu_violation",
                flagger_kwargs={"multiplier": 1.0, "tol_floor": 0.01},
                use_roi=True,
                by_option_type=True,
                requires_wide=False,
                sample_cols=BASE_KEYS + ["mid_price", "bid_price", "ask_price"],
            ),
            SoftRowSpec(
                base_name="pcp_mid_eu_forward",
                flagger=flag_put_call_parity_mid_eu_forward,
                thresholds={"mild": 0.05, "warn": 0.10, "fail": 0.20},
                violation_col="pcp_mid_eu_violation",
                flagger_kwargs={"multiplier": 1.0, "tol_floor": 0.01},
                use_roi=True,
                by_option_type=False,
                requires_wide=True,
                sample_cols=BASE_KEYS
                + [
                    "call_bid_price",
                    "call_mid_price",
                    "call_ask_price",
                    "put_bid_price",
                    "put_mid_price",
                    "put_ask_price",
                ],
            ),
        ]

    if exercise_style == "AM":
        return [
            SoftDatasetSpec(
                base_name="spot_equals_underlying_per_trade_date",
                checker=check_spot_equals_underlying_per_trade_date_am,
                checker_kwargs={"tol_abs": 0.001, "tol_rel": 5e-4},
                thresholds={"mild": 0.001, "warn": 0.01, "fail": 0.05},
                use_roi=False,
            ),
            SoftRowSpec(
                base_name="price_bounds_mid_am",
                flagger=flag_option_bounds_mid_am_spot,
                thresholds={"mild": 0.05, "warn": 0.10, "fail": 0.20},
                violation_col="price_bounds_mid_am_spot_violation",
                flagger_kwargs={"multiplier": 1.0, "tol_floor": 0.01},
                use_roi=True,
                by_option_type=True,
                requires_wide=False,
                sample_cols=BASE_KEYS + ["mid_price", "bid_price", "ask_price"],
            ),
            SoftRowSpec(
                base_name="pcp_bounds_mid_am",
                flagger=flag_put_call_parity_bounds_mid_am,
                thresholds={"mild": 0.05, "warn": 0.10, "fail": 0.20},
                violation_col="pcp_bounds_mid_am_violation",
                flagger_kwargs={"multiplier": 1.0, "tol_floor": 0.01},
                use_roi=True,
                by_option_type=False,
                requires_wide=True,
                sample_cols=BASE_KEYS
                + [
                    "call_bid_price",
                    "call_mid_price",
                    "call_ask_price",
                    "put_bid_price",
                    "put_mid_price",
                    "put_ask_price",
                ],
            ),
        ]

    return []


def get_soft_specs(
    exercise_style: str | None,
    config: object,  # not used now, but kept for future dynamic behavior
) -> list[SoftSpec]:
    specs: list[SoftSpec] = []
    specs.extend(_get_base_soft_specs())
    specs.extend(_get_exercise_soft_specs(exercise_style=exercise_style))
    return specs
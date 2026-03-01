"""Canonical options-chain normalization and validation pipeline."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from volatility_trading.contracts.options_chain import (
    ASK_PRICE,
    BID_PRICE,
    CANONICAL_REQUIRED_COLUMNS,
    DELTA,
    DTE,
    EXPIRY_DATE,
    NUMERIC_COLUMNS,
    OPTION_TYPE,
    STRIKE,
    TRADE_DATE,
)

_VALID_OPTION_TYPES = frozenset({"C", "P"})
_REQUIRED_NUMERIC_COLUMNS = (DTE, STRIKE, DELTA, BID_PRICE, ASK_PRICE)
ValidationMode = Literal["coerce", "strict"]


class OptionsChainAdapterError(ValueError):
    """Raised when options-chain normalization/validation fails."""


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.strip().str.replace(",", "", regex=False),
        errors="coerce",
    )


def _coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str).str.strip(), errors="coerce")


def _canonicalize_option_type(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.upper()
    normalized = normalized.replace(
        {
            "CALL": "C",
            "PUT": "P",
        }
    )
    return normalized


def _set_trade_date_index(
    df: pd.DataFrame,
    *,
    validation_mode: ValidationMode,
) -> pd.DataFrame:
    if TRADE_DATE in df.columns:
        out = df.copy()
        if validation_mode == "coerce":
            out[TRADE_DATE] = _coerce_datetime(out[TRADE_DATE])
        elif not pd.api.types.is_datetime64_any_dtype(out[TRADE_DATE]):
            raise OptionsChainAdapterError(
                f"{TRADE_DATE} must be datetime-like for strict validation"
            )
        out = out.set_index(TRADE_DATE).sort_index()
        out.index.name = TRADE_DATE
        return out

    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        if validation_mode == "coerce":
            out.index = pd.to_datetime(out.index, errors="coerce")
        out.index.name = TRADE_DATE
        return out.sort_index()

    if validation_mode == "strict":
        raise OptionsChainAdapterError(
            f"{TRADE_DATE} must be provided as a datetime-like column or DatetimeIndex "
            "for strict validation"
        )

    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out.index.name = TRADE_DATE
    return out.sort_index()


def _prepare_canonical_df(
    options: pd.DataFrame,
    *,
    validation_mode: ValidationMode,
) -> pd.DataFrame:
    """Prepare canonical dataframe for validation under one mode."""
    df = _set_trade_date_index(options, validation_mode=validation_mode)
    if validation_mode == "strict":
        return df

    if EXPIRY_DATE in df.columns:
        df[EXPIRY_DATE] = _coerce_datetime(df[EXPIRY_DATE])

    if OPTION_TYPE in df.columns:
        df[OPTION_TYPE] = _canonicalize_option_type(df[OPTION_TYPE])

    numeric_cols = [col for col in NUMERIC_COLUMNS if col in df.columns]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].apply(_coerce_numeric)

    return df


def _validate_canonical_df(
    df: pd.DataFrame,
    *,
    adapter_name: str,
    validation_mode: ValidationMode,
) -> None:
    """Validate canonical schema and core data sanity constraints."""
    if df.index.isna().any():
        raise OptionsChainAdapterError(
            f"{adapter_name}: could not parse one or more {TRADE_DATE} values"
        )

    missing_required = [
        col for col in CANONICAL_REQUIRED_COLUMNS if col not in df.columns
    ]
    if missing_required:
        raise OptionsChainAdapterError(
            f"{adapter_name}: missing required canonical columns: {missing_required}"
        )

    if OPTION_TYPE in df.columns:
        bad_option_types = sorted(set(df[OPTION_TYPE].dropna()) - _VALID_OPTION_TYPES)
        if bad_option_types:
            raise OptionsChainAdapterError(
                f"{adapter_name}: {OPTION_TYPE} must be C/P (or call/put). "
                f"Found invalid labels: {bad_option_types}"
            )

    strict_validation = validation_mode == "strict"
    coercion_context = (
        "after coercion" if validation_mode == "coerce" else "during strict validation"
    )
    for required_numeric in _REQUIRED_NUMERIC_COLUMNS:
        if strict_validation and not pd.api.types.is_numeric_dtype(
            df[required_numeric]
        ):
            raise OptionsChainAdapterError(
                f"{adapter_name}: required numeric column '{required_numeric}' must be numeric "
                "for strict validation"
            )
        if df[required_numeric].isna().all():
            raise OptionsChainAdapterError(
                f"{adapter_name}: required numeric column '{required_numeric}' is all-null "
                f"{coercion_context}"
            )

    if strict_validation and not pd.api.types.is_datetime64_any_dtype(df[EXPIRY_DATE]):
        raise OptionsChainAdapterError(
            f"{adapter_name}: {EXPIRY_DATE} must be datetime-like for strict validation"
        )

    expiry_context = (
        "after datetime coercion"
        if validation_mode == "coerce"
        else "during strict validation"
    )
    if df[EXPIRY_DATE].isna().all():
        raise OptionsChainAdapterError(
            f"{adapter_name}: {EXPIRY_DATE} is all-null {expiry_context}"
        )


def validate_options_chain(
    options: pd.DataFrame,
    *,
    adapter_name: str = "unknown",
    validation_mode: ValidationMode = "coerce",
) -> pd.DataFrame:
    """Validate one options-chain dataframe under a chosen validation mode."""
    if options.empty:
        raise OptionsChainAdapterError(f"{adapter_name}: options dataframe is empty")

    df = _prepare_canonical_df(options, validation_mode=validation_mode)
    _validate_canonical_df(
        df,
        adapter_name=adapter_name,
        validation_mode=validation_mode,
    )
    return df


def normalize_and_validate_options_chain(
    options: pd.DataFrame,
    *,
    adapter_name: str = "unknown",
) -> pd.DataFrame:
    """Normalize and validate a canonical options-chain dataframe."""
    return validate_options_chain(
        options,
        adapter_name=adapter_name,
        validation_mode="coerce",
    )


def validate_options_chain_contract(
    options: pd.DataFrame,
    *,
    adapter_name: str = "canonical",
) -> pd.DataFrame:
    """Validate one trusted canonical options-chain dataframe without coercion."""
    return validate_options_chain(
        options,
        adapter_name=adapter_name,
        validation_mode="strict",
    )

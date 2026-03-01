"""Options-chain data adapters and schema validation for backtesting runtime.

This module defines the boundary between arbitrary provider dataframes and the
canonical options-engine contract. Adapters are responsible for:

1. normalizing source-specific columns to canonical names,
2. coercing key fields to stable dtypes, and
3. validating required schema constraints before plan compilation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from volatility_trading.config.options_chain_sources import (
    OPTIONSDX_ALIAS_OVERRIDES,
    ORATS_ALIAS_OVERRIDES,
    YFINANCE_ALIAS_OVERRIDES,
)
from volatility_trading.contracts.options_chain import (
    ASK_PRICE,
    BID_PRICE,
    CANONICAL_ALIAS_FIELDS,
    CANONICAL_COLUMN_SET,
    CANONICAL_REQUIRED_COLUMNS,
    DELTA,
    DTE,
    EXPIRY_DATE,
    NUMERIC_COLUMNS,
    OPTION_TYPE,
    STRIKE,
    TRADE_DATE,
)


class OptionsChainAdapterError(ValueError):
    """Raised when options-chain normalization/validation fails."""


class OptionsChainAdapter(Protocol):
    """Protocol for source-specific options-chain normalization adapters."""

    @property
    def name(self) -> str:
        """Adapter identifier used in validation error messages."""

    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Return source data normalized to canonical engine columns."""


_VALID_OPTION_TYPES = frozenset({"C", "P"})
_REQUIRED_NUMERIC_COLUMNS = (DTE, STRIKE, DELTA, BID_PRICE, ASK_PRICE)


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


def _set_trade_date_index(df: pd.DataFrame) -> pd.DataFrame:
    if TRADE_DATE in df.columns:
        out = df.copy()
        out[TRADE_DATE] = _coerce_datetime(out[TRADE_DATE])
        out = out.set_index(TRADE_DATE).sort_index()
        out.index.name = TRADE_DATE
        return out

    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out.index = pd.to_datetime(out.index, errors="coerce")
        out.index.name = TRADE_DATE
        return out.sort_index()

    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out.index.name = TRADE_DATE
    return out.sort_index()


def _resolve_alias_column(
    *,
    columns: Sequence[str],
    aliases: Sequence[str],
) -> str | None:
    for alias in aliases:
        if alias in columns:
            return alias
    return None


@dataclass(frozen=True)
class AliasOptionsChainAdapter:
    """Adapter based on canonical field aliases.

    ``aliases`` is a mapping from canonical field name to source column aliases.
    The first alias found in the input is selected.
    """

    name: str
    aliases: Mapping[str, Sequence[str]]
    parse_option_type_from_contract_symbol: bool = False

    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            raise OptionsChainAdapterError(
                f"{self.name}: input options dataframe is empty"
            )

        df = raw.copy()
        rename_map: dict[str, str] = {}
        columns = list(df.columns)

        for canonical_name, alias_list in self.aliases.items():
            source_col = _resolve_alias_column(columns=columns, aliases=alias_list)
            if source_col is not None and source_col != canonical_name:
                rename_map[source_col] = canonical_name
        if rename_map:
            df = df.rename(columns=rename_map)

        if (
            self.parse_option_type_from_contract_symbol
            and OPTION_TYPE not in df.columns
            and "contract_symbol" in df.columns
        ):
            symbol = df["contract_symbol"].astype(str).str.upper()
            parsed = symbol.str.extract(r"(\d{6,8})([CP])\d+$", expand=True)
            df[OPTION_TYPE] = parsed.iloc[:, 1]

        return normalize_and_validate_options_chain(df, adapter_name=self.name)


@dataclass(frozen=True)
class ColumnMapOptionsChainAdapter:
    """Generic adapter using a user-provided source-to-canonical column map."""

    source_to_canonical: Mapping[str, str]
    name: str = "column_map"

    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            raise OptionsChainAdapterError(
                f"{self.name}: input options dataframe is empty"
            )

        invalid_targets = [
            canonical
            for canonical in self.source_to_canonical.values()
            if canonical not in CANONICAL_COLUMN_SET and canonical != TRADE_DATE
        ]
        if invalid_targets:
            unique_targets = sorted(set(invalid_targets))
            raise OptionsChainAdapterError(
                f"{self.name}: unsupported canonical targets in source_to_canonical: "
                f"{unique_targets}"
            )

        df = raw.rename(columns=dict(self.source_to_canonical)).copy()
        return normalize_and_validate_options_chain(df, adapter_name=self.name)


@dataclass(frozen=True)
class CanonicalOptionsChainAdapter:
    """Fast-path adapter for trusted canonical inputs.

    This adapter skips alias remapping and numeric coercion. It only enforces
    canonical schema/typing checks. Use it for ETL outputs already normalized
    to the canonical options-chain contract.
    """

    name: str = "canonical"

    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        return validate_options_chain_contract(raw, adapter_name=self.name)


def _unique_aliases(*values: str) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def _build_aliases(
    overrides: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    valid_fields = set(CANONICAL_ALIAS_FIELDS)
    unknown_fields = sorted(set(overrides) - valid_fields)
    if unknown_fields:
        raise ValueError(
            f"unsupported canonical fields in alias overrides: {unknown_fields}"
        )

    aliases: dict[str, tuple[str, ...]] = {}
    for field_name in CANONICAL_ALIAS_FIELDS:
        aliases[field_name] = _unique_aliases(
            field_name, *tuple(overrides.get(field_name, ()))
        )
    return aliases


ORATS_ALIASES = _build_aliases(ORATS_ALIAS_OVERRIDES)
YFINANCE_ALIASES = _build_aliases(YFINANCE_ALIAS_OVERRIDES)
OPTIONSDX_ALIASES = _build_aliases(OPTIONSDX_ALIAS_OVERRIDES)


@dataclass(frozen=True)
class OratsOptionsChainAdapter(AliasOptionsChainAdapter):
    """Built-in adapter for ORATS-like long chain inputs."""

    name: str = "orats"
    aliases: Mapping[str, Sequence[str]] = field(default_factory=lambda: ORATS_ALIASES)


@dataclass(frozen=True)
class YfinanceOptionsChainAdapter(AliasOptionsChainAdapter):
    """Best-effort adapter for yfinance-like option-chain exports.

    Notes:
    - Greeks are not provided by default yfinance chains.
    - Strategies requiring delta-based selection still need a `delta` column
      added upstream (for example via external Greeks computation).
    """

    name: str = "yfinance"
    aliases: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: YFINANCE_ALIASES
    )
    parse_option_type_from_contract_symbol: bool = True


@dataclass(frozen=True)
class OptionsDxOptionsChainAdapter(AliasOptionsChainAdapter):
    """Adapter for cleaned OptionsDX long-format chain data.

    Notes:
    - Input must be long format (`option_type`, not `c_*`/`p_*` columns).
    - Vendor `iv` is mapped to canonical `market_iv`.
    - `open_interest` is optional and may be absent in OptionsDX feeds.
    """

    name: str = "optionsdx"
    aliases: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: OPTIONSDX_ALIASES
    )

    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        lowered = raw.rename(
            columns={col: str(col).strip().lower() for col in raw.columns}
        )
        if OPTION_TYPE not in lowered.columns and any(
            col.startswith(("c_", "p_")) for col in lowered.columns
        ):
            raise OptionsChainAdapterError(
                "optionsdx: wide c_*/p_* columns detected. "
                "Run OptionsDX ETL with reshape='long' before backtesting."
            )
        return super().normalize(lowered)


def _normalize_canonical_df(options: pd.DataFrame) -> pd.DataFrame:
    """Coerce canonical dataframe fields before contract validation."""
    df = _set_trade_date_index(options)

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
    assume_coerced: bool,
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

    coercion_context = "after coercion" if assume_coerced else "during validation"
    for required_numeric in _REQUIRED_NUMERIC_COLUMNS:
        if not assume_coerced and not pd.api.types.is_numeric_dtype(
            df[required_numeric]
        ):
            raise OptionsChainAdapterError(
                f"{adapter_name}: required numeric column '{required_numeric}' must be numeric "
                "for canonical validation"
            )
        if df[required_numeric].isna().all():
            raise OptionsChainAdapterError(
                f"{adapter_name}: required numeric column '{required_numeric}' is all-null "
                f"{coercion_context}"
            )

    if not assume_coerced and not pd.api.types.is_datetime64_any_dtype(df[EXPIRY_DATE]):
        raise OptionsChainAdapterError(
            f"{adapter_name}: {EXPIRY_DATE} must be datetime-like for canonical validation"
        )

    expiry_context = (
        "after datetime coercion" if assume_coerced else "during validation"
    )
    if df[EXPIRY_DATE].isna().all():
        raise OptionsChainAdapterError(
            f"{adapter_name}: {EXPIRY_DATE} is all-null {expiry_context}"
        )


def validate_options_chain_contract(
    options: pd.DataFrame,
    *,
    adapter_name: str = "canonical",
) -> pd.DataFrame:
    """Validate a trusted canonical options-chain dataframe without coercion."""
    if options.empty:
        raise OptionsChainAdapterError(f"{adapter_name}: options dataframe is empty")

    df = _set_trade_date_index(options)
    _validate_canonical_df(df, adapter_name=adapter_name, assume_coerced=False)
    return df


def normalize_and_validate_options_chain(
    options: pd.DataFrame,
    *,
    adapter_name: str = "unknown",
) -> pd.DataFrame:
    """Normalize and validate a canonical options-chain dataframe."""
    if options.empty:
        raise OptionsChainAdapterError(f"{adapter_name}: options dataframe is empty")

    df = _normalize_canonical_df(options)
    _validate_canonical_df(df, adapter_name=adapter_name, assume_coerced=True)
    return df


def normalize_options_chain(
    options: pd.DataFrame,
    *,
    adapter: OptionsChainAdapter | None,
) -> pd.DataFrame:
    """Apply adapter boundary before options execution plan compilation."""
    active_adapter = adapter or OratsOptionsChainAdapter()
    return active_adapter.normalize(options)

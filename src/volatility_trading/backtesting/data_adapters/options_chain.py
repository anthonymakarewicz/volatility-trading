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

CANONICAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "expiry_date",
    "dte",
    "option_type",
    "strike",
    "delta",
    "bid_price",
    "ask_price",
)

CANONICAL_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "gamma",
    "vega",
    "theta",
    "spot_price",
    "market_iv",
    "model_iv",
    "yte",
    "open_interest",
    "volume",
)

CANONICAL_COLUMN_SET: frozenset[str] = frozenset(
    CANONICAL_REQUIRED_COLUMNS + CANONICAL_OPTIONAL_COLUMNS
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "dte",
    "strike",
    "delta",
    "gamma",
    "vega",
    "theta",
    "bid_price",
    "ask_price",
    "spot_price",
    "market_iv",
    "model_iv",
    "yte",
    "open_interest",
    "volume",
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
    if "trade_date" in df.columns:
        out = df.copy()
        out["trade_date"] = _coerce_datetime(out["trade_date"])
        out = out.set_index("trade_date").sort_index()
        out.index.name = "trade_date"
        return out

    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out.index = pd.to_datetime(out.index, errors="coerce")
        out.index.name = "trade_date"
        return out.sort_index()

    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out.index.name = "trade_date"
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
            and "option_type" not in df.columns
            and "contract_symbol" in df.columns
        ):
            symbol = df["contract_symbol"].astype(str).str.upper()
            parsed = symbol.str.extract(r"(\d{6,8})([CP])\d+$", expand=True)
            df["option_type"] = parsed.iloc[:, 1]

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
            if canonical not in CANONICAL_COLUMN_SET and canonical != "trade_date"
        ]
        if invalid_targets:
            unique_targets = sorted(set(invalid_targets))
            raise OptionsChainAdapterError(
                f"{self.name}: unsupported canonical targets in source_to_canonical: "
                f"{unique_targets}"
            )

        df = raw.rename(columns=dict(self.source_to_canonical)).copy()
        return normalize_and_validate_options_chain(df, adapter_name=self.name)


CANONICAL_ALIAS_FIELDS: tuple[str, ...] = (
    "trade_date",
    *CANONICAL_REQUIRED_COLUMNS,
    *CANONICAL_OPTIONAL_COLUMNS,
)


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


ORATS_ALIAS_OVERRIDES: dict[str, tuple[str, ...]] = {
    "trade_date": ("date", "quote_date"),
    "expiry_date": ("expiry", "expire_date"),
    "bid_price": ("bid",),
    "ask_price": ("ask",),
    "spot_price": ("underlying_last", "underlying_price"),
    "market_iv": ("market_iv", "mid_iv", "iv", "smoothed_iv"),
    "model_iv": ("model_iv", "smoothed_iv"),
    "open_interest": ("oi",),
}

YFINANCE_ALIAS_OVERRIDES: dict[str, tuple[str, ...]] = {
    "trade_date": ("quote_date", "date", "last_trade_date"),
    "expiry_date": ("expiration", "expiry", "expiration_date"),
    "option_type": ("type", "option_type_label"),
    "bid_price": ("bid",),
    "ask_price": ("ask",),
    "spot_price": ("underlying_price", "underlying_last"),
    "market_iv": ("implied_volatility", "impliedVolatility", "iv", "smoothed_iv"),
    "open_interest": ("openInterest",),
}

OPTIONSDX_ALIAS_OVERRIDES: dict[str, tuple[str, ...]] = {
    "trade_date": ("date", "quote_date", "quote_readtime"),
    "expiry_date": ("expiry", "expire_date"),
    "bid_price": ("bid",),
    "ask_price": ("ask",),
    "spot_price": ("underlying_last", "underlying_price"),
    # OptionsDX exposes market IV; map to canonical market_iv.
    "market_iv": ("mid_iv", "iv", "smoothed_iv"),
    "open_interest": ("oi",),
}

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
        if "option_type" not in lowered.columns and any(
            col.startswith(("c_", "p_")) for col in lowered.columns
        ):
            raise OptionsChainAdapterError(
                "optionsdx: wide c_*/p_* columns detected. "
                "Run OptionsDX ETL with reshape='long' before backtesting."
            )  # TODO: Issue warnig and maybe reshape to long format here
        return super().normalize(lowered)


def normalize_and_validate_options_chain(
    options: pd.DataFrame,
    *,
    adapter_name: str = "unknown",
) -> pd.DataFrame:
    """Normalize and validate a canonical options-chain dataframe."""
    if options.empty:
        raise OptionsChainAdapterError(f"{adapter_name}: options dataframe is empty")

    df = _set_trade_date_index(options)

    if df.index.isna().any():
        raise OptionsChainAdapterError(
            f"{adapter_name}: could not parse one or more trade_date values"
        )

    if "expiry_date" in df.columns:
        df = df.copy()
        df["expiry_date"] = _coerce_datetime(df["expiry_date"])

    if "option_type" in df.columns:
        df = df.copy()
        df["option_type"] = _canonicalize_option_type(df["option_type"])

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df = df.copy()
            df[col] = _coerce_numeric(df[col])

    missing_required = [
        col for col in CANONICAL_REQUIRED_COLUMNS if col not in df.columns
    ]
    if missing_required:
        raise OptionsChainAdapterError(
            f"{adapter_name}: missing required canonical columns: {missing_required}"
        )

    bad_option_types = sorted(set(df["option_type"].dropna()) - {"C", "P"})
    if bad_option_types:
        raise OptionsChainAdapterError(
            f"{adapter_name}: option_type must be C/P (or call/put). "
            f"Found invalid labels: {bad_option_types}"
        )

    for required_numeric in ("dte", "strike", "delta", "bid_price", "ask_price"):
        if df[required_numeric].isna().all():
            raise OptionsChainAdapterError(
                f"{adapter_name}: required numeric column '{required_numeric}' is all-null "
                "after coercion"
            )

    if df["expiry_date"].isna().all():
        raise OptionsChainAdapterError(
            f"{adapter_name}: expiry_date is all-null after datetime coercion"
        )

    return df


def normalize_options_chain(
    options: pd.DataFrame,
    *,
    adapter: OptionsChainAdapter | None,
) -> pd.DataFrame:
    """Apply adapter boundary before options execution plan compilation."""
    active_adapter = adapter or OratsOptionsChainAdapter()
    return active_adapter.normalize(options)

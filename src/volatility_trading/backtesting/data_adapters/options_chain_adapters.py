"""Options-chain adapters for source-to-canonical schema normalization."""

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
    CANONICAL_ALIAS_FIELDS,
    CANONICAL_COLUMN_SET,
    OPTION_TYPE,
    TRADE_DATE,
)

from .options_chain_pipeline import (
    OptionsChainAdapterError,
    normalize_and_validate_options_chain,
    validate_options_chain_contract,
)


class OptionsChainAdapter(Protocol):
    """Protocol for source-specific options-chain normalization adapters."""

    @property
    def name(self) -> str:
        """Adapter identifier used in validation error messages."""

    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Return source data normalized to canonical engine columns."""


def _resolve_alias_column(
    *,
    columns: Sequence[str],
    aliases: Sequence[str],
) -> str | None:
    for alias in aliases:
        if alias in columns:
            return alias
    return None


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
    """Fast-path adapter for trusted canonical inputs."""

    name: str = "canonical"

    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        return validate_options_chain_contract(raw, adapter_name=self.name)


@dataclass(frozen=True)
class OratsOptionsChainAdapter(AliasOptionsChainAdapter):
    """Built-in adapter for ORATS-like long chain inputs."""

    name: str = "orats"
    aliases: Mapping[str, Sequence[str]] = field(default_factory=lambda: ORATS_ALIASES)


@dataclass(frozen=True)
class YfinanceOptionsChainAdapter(AliasOptionsChainAdapter):
    """Best-effort adapter for yfinance-like option-chain exports."""

    name: str = "yfinance"
    aliases: Mapping[str, Sequence[str]] = field(
        default_factory=lambda: YFINANCE_ALIASES
    )
    parse_option_type_from_contract_symbol: bool = True


@dataclass(frozen=True)
class OptionsDxOptionsChainAdapter(AliasOptionsChainAdapter):
    """Adapter for cleaned OptionsDX long-format chain data."""

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


def normalize_options_chain(
    options: pd.DataFrame,
    *,
    adapter: OptionsChainAdapter,
) -> pd.DataFrame:
    """Apply adapter boundary before options execution plan compilation."""
    return adapter.normalize(options)

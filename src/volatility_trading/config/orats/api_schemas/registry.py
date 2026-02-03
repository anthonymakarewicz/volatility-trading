"""
ORATS API schema specs (raw JSON -> intermediate parquet).

These specs are meant to support mechanical normalization:
- cast vendor fields to stable dtypes (pre-rename)
- parse vendor date/datetime fields (pre-rename)
- rename vendor -> canonical
- select canonical columns to keep in intermediate
- bounds:
    * bounds_drop_canonical: out-of-bounds => drop row (structural validity)
    * bounds_null_canonical: out-of-bounds => set to null

The API schemas are stored by logical endpoint name.
"""

from __future__ import annotations

from typing import Final

from ..schema_spec import OratsSchemaSpec
from .hvs import HVS_SCHEMA_SPEC
from .monies_implied import MONIES_IMPLIED_SCHEMA
from .summaries import SUMMARIES_SCHEMA


# TODO: Implement schema specs for these endpoints (downloaded in RAW)
UNSUPPORTED_ENDPOINTS: tuple[str, ...] = ("cores", "splits", "ivrank")


API_SCHEMAS: Final[dict[str, OratsSchemaSpec]] = {
    "monies_implied": MONIES_IMPLIED_SCHEMA,
    "summaries": SUMMARIES_SCHEMA,
    "hvs": HVS_SCHEMA_SPEC,
}


def get_schema_spec(endpoint: str) -> OratsSchemaSpec:
    """Return the schema spec for a supported ORATS API endpoint.

    Raises
    ------
    KeyError
        If the endpoint is unknown.
    """
    try:
        return API_SCHEMAS[endpoint]
    except KeyError as e:
        supported = ", ".join(sorted(API_SCHEMAS.keys()))
        raise KeyError(
            f"Unknown ORATS API schema for endpoint '{endpoint}'. "
            f"Supported: {supported}"
        ) from e
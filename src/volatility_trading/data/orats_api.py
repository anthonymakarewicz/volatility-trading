from __future__ import annotations

import json
import polars as pl


def orats_response_to_polars(response_text: str) -> pl.DataFrame:
    """
    Convert ORATS API JSON response (string) to a Polars DataFrame.

    ORATS responses typically look like:
      {"data":[{...},{...}], "header":{...}}  (header optional)

    We also defensively strip whitespace from keys because some payloads
    contain leading spaces in field names (e.g. " expirDate").
    """
    payload = json.loads(response_text)

    data = payload.get("data", [])
    if not data:
        # Could also inspect payload.get("message") / payload.get("error")
        return pl.DataFrame()

    # Strip whitespace in keys (handles " expirDate", " tradeDate", etc.)
    cleaned = [{k.strip(): v for k, v in row.items()} for row in data]

    df = pl.DataFrame(cleaned)

    # Optional: parse date columns if present
    for c in ("tradeDate", "expirDate"):
        if c in df.columns:
            df = df.with_columns(
                pl.col(c).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
            )

    return df
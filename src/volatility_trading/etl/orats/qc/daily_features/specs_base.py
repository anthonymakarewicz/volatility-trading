from __future__ import annotations

from volatility_trading.etl.orats.processed.daily_features.config import (
    DAILY_FEATURES_CORE_COLUMNS,
)


BASE_KEYS = [
    "ticker",
    "trade_date",
]

IV_COLUMNS = tuple(
    col for col in DAILY_FEATURES_CORE_COLUMNS if col.startswith("iv_")
)
HV_COLUMNS = tuple(
    col for col in DAILY_FEATURES_CORE_COLUMNS if col.startswith("hv_")
)

INFO_COLUMNS = tuple(c for c in DAILY_FEATURES_CORE_COLUMNS if c not in BASE_KEYS)

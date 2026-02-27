"""Typed dataset bundles consumed by the backtesting runtime."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class OptionsBacktestDataBundle:
    """Typed input datasets consumed by options backtesting runtime."""

    options: pd.DataFrame
    features: pd.DataFrame | None = None
    hedge: pd.Series | pd.DataFrame | None = None
    fallback_iv_feature_col: str = "iv_atm"

"""Dataset I/O helpers for processed research datasets.

This package exposes thin convenience wrappers to locate, scan, read, and
reshape processed ORATS panels as well as external market-feed datasets.
"""

from .daily_features import (
    daily_features_path,
    join_daily_features,
    read_daily_features,
    scan_daily_features,
)
from .fred import (
    fred_domain_path,
    fred_market_path,
    fred_rates_path,
    read_fred_domain,
    read_fred_market,
    read_fred_rates,
    scan_fred_domain,
    scan_fred_market,
    scan_fred_rates,
)
from .options_chain import (
    options_chain_long_to_wide,
    options_chain_path,
    options_chain_wide_to_long,
    read_options_chain,
    scan_options_chain,
)
from .yfinance import (
    read_yfinance_time_series,
    scan_yfinance_time_series,
    yfinance_time_series_path,
)

__all__ = [
    "daily_features_path",
    "fred_domain_path",
    "fred_market_path",
    "fred_rates_path",
    "join_daily_features",
    "options_chain_long_to_wide",
    "options_chain_path",
    "options_chain_wide_to_long",
    "read_daily_features",
    "read_fred_domain",
    "read_fred_market",
    "read_fred_rates",
    "read_options_chain",
    "read_yfinance_time_series",
    "scan_daily_features",
    "scan_fred_domain",
    "scan_fred_market",
    "scan_fred_rates",
    "scan_options_chain",
    "scan_yfinance_time_series",
    "yfinance_time_series_path",
]

"""Canonical filesystem paths used across the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ----- Base data roots -----
DATA_ROOT = PROJECT_ROOT / "data"
DATA_RAW = DATA_ROOT / "raw"
DATA_INTER = DATA_ROOT / "intermediate"
DATA_PROC = DATA_ROOT / "processed"

# ----- Reports roots -----
REPORTS_ROOT = PROJECT_ROOT / "reports"
BACKTEST_REPORTS_ROOT = REPORTS_ROOT / "backtests"

# ----- Options – raw -----
RAW_ORATS = DATA_RAW / "orats"  # provider root
RAW_ORATS_FTP = RAW_ORATS / "ftp"
RAW_ORATS_API = RAW_ORATS / "api"

RAW_OPTIONSDX = DATA_RAW / "optionsdx"

RAW_YFINANCE = DATA_RAW / "yfinance"
RAW_YFINANCE_TIME_SERIES = RAW_YFINANCE / "time_series"

RAW_FRED = DATA_RAW / "fred"
RAW_FRED_RATES = RAW_FRED / "rates"
RAW_FRED_MARKET = RAW_FRED / "market"

# ----- Options – intermediate -----
INTER_ORATS = DATA_INTER / "orats"  # provider root
INTER_ORATS_FTP = INTER_ORATS / "ftp" / "strikes"
INTER_ORATS_API = INTER_ORATS / "api"

INTER_OPTIONSDX = DATA_INTER / "optionsdx"

# ----- Processed roots by source -----
PROC_ORATS = DATA_PROC / "orats"
PROC_ORATS_OPTIONS_CHAIN = PROC_ORATS / "options_chain"
PROC_ORATS_DAILY_FEATURES = PROC_ORATS / "daily_features"

PROC_OPTIONSDX = DATA_PROC / "optionsdx"
PROC_OPTIONSDX_OPTIONS_CHAIN = PROC_OPTIONSDX / "options_chain"
PROC_OPTIONSDX_DAILY_FEATURES = PROC_OPTIONSDX / "daily_features"

PROC_YFINANCE = DATA_PROC / "yfinance"
PROC_YFINANCE_TIME_SERIES = PROC_YFINANCE / "time_series"

PROC_FRED = DATA_PROC / "fred"
PROC_FRED_RATES = PROC_FRED / "rates"
PROC_FRED_MARKET = PROC_FRED / "market"

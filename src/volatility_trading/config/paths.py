from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

DATA_ROOT = ROOT / "data"
DATA_RAW = DATA_ROOT / "raw"
DATA_INTER = DATA_ROOT / "intermediate"
DATA_PROC = DATA_ROOT / "processed"

# ----- Options – raw -----
RAW_ORATS = DATA_RAW / "orats"  # provider root
RAW_ORATS_FTP = RAW_ORATS / "ftp"
RAW_ORATS_API = RAW_ORATS / "api"

RAW_OPTIONSDX = DATA_RAW / "optionsdx"

# ----- Options – intermediate -----
INTER_ORATS = DATA_INTER / "orats"  # provider root

# Legacy (pre-split) location kept for backward compatibility
INTER_ORATS_BY_TICKER = INTER_ORATS / "by_ticker"

# Preferred split layout
INTER_ORATS_FTP = INTER_ORATS / "ftp"
INTER_ORATS_FTP_BY_TICKER = INTER_ORATS_FTP / "by_ticker"

INTER_ORATS_API = INTER_ORATS / "api"
INTER_ORATS_API_BY_TICKER = INTER_ORATS_API / "by_ticker"

INTER_OPTIONSDX = DATA_INTER / "optionsdx"

# ----- Options – processed -----
PROC_ORATS = DATA_PROC / "orats"  # provider root

# Suggested (optional) structure for processed outputs
PROC_ORATS_PANELS = PROC_ORATS / "panels"
PROC_ORATS_FEATURES = PROC_ORATS / "features"

PROC_OPTIONSDX = DATA_PROC / "optionsdx"
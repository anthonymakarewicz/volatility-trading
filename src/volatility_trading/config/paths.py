from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# ----- Base data roots -----
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
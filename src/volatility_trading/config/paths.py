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

# Preferred split layout
INTER_ORATS_FTP = INTER_ORATS / "ftp" / "strikes"
INTER_ORATS_API = INTER_ORATS / "api"

INTER_OPTIONSDX = DATA_INTER / "optionsdx"

# ----- Options – processed -----
PROC_ORATS = DATA_PROC / "orats"  # provider root

# Suggested (optional) structure for processed outputs
PROC_ORATS_PANELS = PROC_ORATS / "panels"
PROC_ORATS_FEATURES = PROC_ORATS / "features"

PROC_OPTIONSDX = DATA_PROC / "optionsdx"
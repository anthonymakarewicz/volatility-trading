from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] # project root

DATA_ROOT = ROOT / "data"
DATA_RAW = DATA_ROOT / "raw"
DATA_INTER = DATA_ROOT / "intermediate"
DATA_PROC = DATA_ROOT / "processed"

# Options – raw
RAW_ORATS = DATA_RAW / "orats"
RAW_OPTIONSDX = DATA_RAW / "optionsdx"

# Options – intermediate
INTER_ORATS = DATA_INTER / "orats"
INTER_OPTIONSDX = DATA_INTER / "optionsdx"

# Options – processed
PROC_ORATS = DATA_PROC / "orats"
PROC_OPTIONSDX = DATA_PROC / "optionsdx"


def raw(asset_class: str, source: str) -> Path:
    return DATA_ROOT / "raw" / asset_class / source

def processed(asset_class: str, source: str) -> Path:
    return DATA_ROOT / "processed" / asset_class / source

def interim(asset_class: str, source: str) -> Path:
    return DATA_ROOT / "interim" / asset_class / source
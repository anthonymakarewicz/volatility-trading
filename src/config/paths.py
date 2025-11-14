from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTER = ROOT / "data" / "intermediate"
DATA_PROC = ROOT / "data" / "processed"
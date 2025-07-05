from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "raw"
INTERMEDIATE_PATH = BASE_DIR / "data" / "intermediate"
PROCESSED_PATH = BASE_DIR / "data" / "processed"
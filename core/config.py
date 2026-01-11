from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DB_PATH = PROCESSED_DIR / "data.duckdb"
SOURCES_YAML = BASE_DIR / "data" / "sources.yml"

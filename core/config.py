from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "raw"
DB_PATH = BASE_DIR / "data" / "processed" / "cobb_douglas.duckdb"
SOURCES_YAML = BASE_DIR / "data" / "sources.yml"

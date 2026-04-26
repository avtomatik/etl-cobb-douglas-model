from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DUCKDB_PATH = BASE_DIR / "data" / "processed" / "cobb_douglas.duckdb"
SOURCES_PATH = BASE_DIR / "data" / "sources.yml"

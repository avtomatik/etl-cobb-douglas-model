from pathlib import Path

import yaml

from core.data import duckdb_connection
from core.paths import SOURCES_PATH


def load_config():
    with SOURCES_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["sources"]


def ingest_zip_json(*, table_name: str, cfg: dict) -> None:
    parquet_path = Path(cfg["path"])
    columns = ", ".join(cfg["columns"])

    with duckdb_connection() as con:
        con.execute("CREATE SCHEMA IF NOT EXISTS raw;")

        con.execute(
            f"""
            CREATE OR REPLACE TABLE raw.{table_name} AS
            SELECT {columns}
            FROM read_parquet('{parquet_path}');
            """
        )

    print(f"Ingested raw.{table_name}")


def main():
    cfgs = load_config()

    for key, value in cfgs.items():
        ingest_zip_json(table_name=key, cfg=value)


if __name__ == "__main__":
    main()

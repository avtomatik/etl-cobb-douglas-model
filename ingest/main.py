import tempfile
import zipfile
from pathlib import Path

import yaml

from core.config import SOURCES_YAML
from core.data import duckdb_connection


def load_config():
    with SOURCES_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["sources"]


def ingest_zip_json(*, table_name: str, cfg: dict) -> None:
    zip_path = Path(cfg["path"])
    data_file = cfg["data_file"]
    columns = ", ".join(cfg["columns"])

    with duckdb_connection() as con:
        con.execute("CREATE SCHEMA IF NOT EXISTS raw;")

        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(data_file) as src:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".json.gz"
                ) as tmp_file:
                    tmp_file.write(src.read())
                    tmp_file_path = Path(tmp_file.name)

        con.execute(
            f"""
            CREATE OR REPLACE TABLE raw.{table_name} AS
            SELECT {columns}
            FROM read_json('{tmp_file_path}');
            """
        )

        tmp_file_path.unlink()

    print(f"Ingested raw.{table_name}")


def main():
    cfgs = load_config()

    for key, value in cfgs.items():
        ingest_zip_json(table_name=key, cfg=value)


if __name__ == "__main__":
    main()

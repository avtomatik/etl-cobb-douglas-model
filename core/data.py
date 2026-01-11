from contextlib import contextmanager

import duckdb

from core.config import DB_PATH


@contextmanager
def duckdb_connection(*, read_only: bool = False):
    con = duckdb.connect(
        str(DB_PATH),
        read_only=read_only,
    )
    try:
        yield con
    finally:
        con.close()

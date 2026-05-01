from contextlib import contextmanager

import duckdb

from core.paths import DUCKDB_PATH


@contextmanager
def duckdb_connection(*, read_only: bool = False):
    con = duckdb.connect(
        str(DUCKDB_PATH),
        read_only=read_only,
    )
    try:
        yield con
    finally:
        con.close()

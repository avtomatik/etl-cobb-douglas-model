from core.data import duckdb_connection

from viz.constants import PLOT_CONTRACT_COLUMNS
from viz.plot import plot_cobb_douglas
from viz.validation import validate_columns


def main():
    with duckdb_connection() as con:
        df = con.execute(
            """
            SELECT
                *
            FROM
                cobb_douglas_series
            ORDER BY
                period
            """
        ).fetchdf()

        validate_columns(df, PLOT_CONTRACT_COLUMNS)

        alpha, scale = con.execute(
            """
            SELECT
                alpha,
                scale
            FROM
                cobb_douglas_estimates
            LIMIT
                1
            """
        ).fetchone()

    plot_cobb_douglas(df, alpha, scale)


if __name__ == "__main__":
    main()

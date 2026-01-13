from core.data import duckdb_connection
from viz.plot import plot_cobb_douglas


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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.data import duckdb_connection


def fetch_data_from_duckdb(query: str) -> pd.DataFrame:
    """
    Fetch data from DuckDB using the provided SQL query and return a pandas DataFrame.

    Parameters
    ----------
    query : str
        The SQL query to fetch data from the DuckDB database.

    Returns
    -------
    pd.DataFrame
        The resulting data in a pandas DataFrame.
    """
    with duckdb_connection() as con:
        df = con.execute(query).fetchdf()
    return df


def combine_cobb_douglas() -> pd.DataFrame:
    """
    Refactor the combine_cobb_douglas() function to use pure SQL for data combination.

    Returns
    -------
    pd.DataFrame
        DataFrame containing combined data for Capital, Labor, and Product.
    """
    sql_query = """
    WITH capital_data AS (
        SELECT period, value AS capital
        FROM raw.usa_cobb_douglas
        WHERE series_id = 'CDT2S4'
    ),
    labor_data AS (
        SELECT period, value AS labor
        FROM raw.usa_cobb_douglas
        WHERE series_id = 'CDT3S1'
    ),
    product_data AS (
        SELECT period, value AS product
        FROM raw.uscb
        WHERE series_id = 'J0014'
    ),
    product_nber_data AS (
        SELECT period, value AS product_nber
        FROM raw.uscb
        WHERE series_id = 'J0013'
    ),
    product_rev_data AS (
        SELECT period, value AS product_rev
        FROM raw.douglas
        WHERE series_id = 'DT24AS01'
    )

    SELECT
        capital_data.period,
        capital_data.capital,
        labor_data.labor,
        product_data.product,
        product_nber_data.product_nber,
        product_rev_data.product_rev
    FROM
        capital_data
    JOIN labor_data ON capital_data.period = labor_data.period
    JOIN product_data ON capital_data.period = product_data.period
    LEFT JOIN product_nber_data ON capital_data.period = product_nber_data.period
    LEFT JOIN product_rev_data ON capital_data.period = product_rev_data.period
    ORDER BY capital_data.period;
    """

    sql_query = """
    WITH capital_data AS (
        SELECT period, value AS capital
        FROM raw.usa_cobb_douglas
        WHERE series_id = 'CDT2S4'
    ),
    labor_data AS (
        SELECT period, value AS labor
        FROM raw.usa_cobb_douglas
        WHERE series_id = 'CDT3S1'
    ),
    product_data AS (
        SELECT period, value AS product
        FROM raw.uscb
        WHERE series_id = 'J0014'
    )

    SELECT
        capital_data.period,
        capital_data.capital,
        labor_data.labor,
        product_data.product
    FROM
        capital_data
    JOIN labor_data ON capital_data.period = labor_data.period
    JOIN product_data ON capital_data.period = product_data.period
    ORDER BY capital_data.period;
    """

    return fetch_data_from_duckdb(sql_query)


def transform_cobb_douglas(
    df: pd.DataFrame, year_base: int
) -> tuple[pd.DataFrame, tuple[float]]:
    """
    ================== =================================
    df.index           Period
    df.iloc[:, 0]      Capital
    df.iloc[:, 1]      Labor
    df.iloc[:, 2]      Product
    ================== =================================
    """
    df = df.div(df.loc[year_base, :])
    # =========================================================================
    # Labor Capital Intensity
    # =========================================================================
    df["lab_cap_int"] = df.iloc[:, 0].div(df.iloc[:, 1])
    # =========================================================================
    # Labor Productivity
    # =========================================================================
    df["lab_product"] = df.iloc[:, 2].div(df.iloc[:, 1])
    # =========================================================================
    # Original: k=0.25, b=1.01
    # =========================================================================
    k, b = np.polyfit(
        np.log(df.iloc[:, -2].astype(float)),
        np.log(df.iloc[:, -1].astype(float)),
        deg=1,
    )
    # =========================================================================
    # Scipy Signal Median Filter, Non-Linear Low-Pass Filter
    # =========================================================================
    # =========================================================================
    # k, b = np.polyfit(
    #     np.log(signal.medfilt(df.iloc[:, -2])),
    #     np.log(signal.medfilt(df.iloc[:, -1])),
    #     deg=1
    # )
    # =========================================================================
    # =========================================================================
    # Description
    # =========================================================================
    df["cap_to_lab"] = df.iloc[:, 1].div(df.iloc[:, 0])
    # =========================================================================
    # Fixed Assets Turnover
    # =========================================================================
    df["c_turnover"] = df.iloc[:, 2].div(df.iloc[:, 0])
    # =========================================================================
    # Product Trend Line=3 Year Moving Average
    # =========================================================================
    df["prod_roll"] = df.iloc[:, 2].rolling(3, center=True).mean()
    df["prod_roll_sub"] = df.iloc[:, 2].sub(df.iloc[:, -1])
    # =========================================================================
    # Computed Product
    # =========================================================================
    df["prod_comp"] = (
        df.iloc[:, 0].pow(k).mul(df.iloc[:, 1].pow(1 - k)).mul(np.exp(b))
    )
    # =========================================================================
    # Computed Product Trend Line=3 Year Moving Average
    # =========================================================================
    df["prod_comp_roll"] = df.iloc[:, -1].rolling(3, center=True).mean()
    df["prod_comp_roll_sub"] = df.iloc[:, -2].sub(df.iloc[:, -1])
    # =========================================================================
    #     print(f"R**2: {r2_score(df.iloc[:, 2], df.iloc[:, 3]):,.4f}")
    #     print(df.iloc[:, 3].div(df.iloc[:, 2]).sub(1).abs().mean())
    # =========================================================================
    return df, (k, np.exp(b))


def lab_productivity(
    array: np.array, k: float = 0.25, b: float = 1.01
) -> np.array:
    return np.multiply(np.power(array, -k), b)


def cap_productivity(
    array: np.array, k: float = 0.25, b: float = 1.01
) -> np.array:
    return np.multiply(np.power(array, 1 - k), b)


def plot_cobb_douglas(
    df: pd.DataFrame, params: tuple[float], mapping: dict
) -> None:
    """
    Cobb--Douglas Algorithm as per C.W. Cobb, P.H. Douglas. A Theory of Production, 1928;
    """
    assert df.shape[1] == 12

    plt.figure(1)
    plt.semilogy(
        df.iloc[:, range(3)],
        label=[
            "Fixed Capital",
            "Labor Force",
            "Physical Product",
        ],
    )
    plt.xlabel("Period")
    plt.ylabel("Indexes")
    plt.title(mapping["fg_a"].format(*df.index[[0, -1]], mapping["year_base"]))
    plt.grid()
    plt.legend()

    plt.figure(2)
    plt.semilogy(
        df.iloc[:, [2, 9]],
        label=[
            "Actual Product",
            "Computed Product, $P' = {:,.4f}L^{{{:,.4f}}}C^{{{:,.4f}}}$".format(
                params[1],
                1 - params[0],
                params[0],
            ),
        ],
    )
    plt.xlabel("Period")
    plt.ylabel("Production")
    plt.title(mapping["fg_b"].format(*df.index[[0, -1]], mapping["year_base"]))
    plt.grid()
    plt.legend()

    plt.figure(3)
    plt.plot(
        df.iloc[:, [8, 11]],
        label=[
            "Deviations of $P$",
            "Deviations of $P'$",
            # =================================================================
            # TODO: ls=['solid','dashed',]
            # =================================================================
        ],
    )
    plt.xlabel("Period")
    plt.ylabel("Percentage Deviation")
    plt.title(mapping["fg_c"])
    plt.grid()
    plt.legend()

    plt.figure(4)
    plt.plot(df.iloc[:, 9].div(df.iloc[:, 2]).sub(1))
    plt.xlabel("Period")
    plt.ylabel("Percentage Deviation")
    plt.title(mapping["fg_d"].format(*df.index[[0, -1]]))
    plt.grid()
    plt.figure(5, figsize=(5, 8))
    plt.scatter(df.iloc[:, 5], df.iloc[:, 4])
    plt.scatter(df.iloc[:, 5], df.iloc[:, 6])
    lc = np.arange(0.2, 1.0, 0.005)
    plt.plot(
        lc, lab_productivity(lc, *params), label="$\\frac{3}{4}\\frac{P}{L}$"
    )
    plt.plot(
        lc, cap_productivity(lc, *params), label="$\\frac{1}{4}\\frac{P}{C}$"
    )
    plt.xlabel("$\\frac{L}{C}$")
    plt.ylabel("Indexes")
    plt.title(mapping["fg_e"])
    plt.grid()
    plt.legend()

    plt.show()


def get_fig_map(year_base: int = 1899) -> dict[str, str]:
    return {
        "fg_a": f"Chart I Progress in Manufacturing {{}}$-${{}} ({year_base}=100)",
        "fg_b": f"Chart II Theoretical and Actual Curves of Production {{}}$-${{}} ({year_base}=100)",
        "fg_c": "Chart III Percentage Deviations of $P$ and $P'$ from Their Trend Lines\nTrend Lines=3 Year Moving Average",
        "fg_d": "Chart IV Percentage Deviations of Computed from Actual Product {}$-${}",
        "fg_e": "Chart V Relative Final Productivities of Labor and Capital",
        "year_base": year_base,
    }


def main():
    YEAR_BASE = 1899

    df, params = (
        combine_cobb_douglas()
        .set_index("period")
        .pipe(transform_cobb_douglas, year_base=YEAR_BASE)
    )

    plot_cobb_douglas(df, params, get_fig_map(YEAR_BASE))


if __name__ == "__main__":
    main()

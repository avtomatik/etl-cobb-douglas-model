import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.data import duckdb_connection


def fetch_dataframe(sql: str) -> pd.DataFrame:
    """
    Execute a SQL query against DuckDB and return a DataFrame.
    """
    with duckdb_connection() as con:
        return con.execute(sql).fetchdf()


def load_cobb_douglas_inputs() -> pd.DataFrame:
    """
    Load capital, labor, and product series aligned by period.

    Returns
    -------
    pd.DataFrame
        Columns: [period, capital, labor, product]
    """
    sql = """
    WITH
        capital AS (
            SELECT period, value AS capital
            FROM raw.usa_cobb_douglas
            WHERE series_id = 'CDT2S4'
        ),
        labor AS (
            SELECT period, value AS labor
            FROM raw.usa_cobb_douglas
            WHERE series_id = 'CDT3S1'
        ),
        product AS (
            SELECT period, value AS product
            FROM raw.uscb
            WHERE series_id = 'J0014'
        )
    SELECT
        capital.period,
        capital.capital,
        labor.labor,
        product.product
    FROM capital
    JOIN labor   USING (period)
    JOIN product USING (period)
    ORDER BY period;
    """
    return fetch_dataframe(sql)


def estimate_cobb_douglas(
    df: pd.DataFrame,
    base_year: int,
) -> tuple[pd.DataFrame, float, float]:
    """
    Normalize data, compute productivity metrics, and estimate
    Cobb–Douglas parameters.

    Parameters
    ----------
    df : DataFrame
        Indexed by period with columns [capital, labor, product]
    base_year : int
        Normalization base year (index = 100)

    Returns
    -------
    df : DataFrame
        Enriched dataset
    alpha : float
        Capital elasticity (k)
    scale : float
        Total factor productivity (A)
    """
    df = df.copy()

    # =========================================================================
    # Normalize indexes (base_year = 100)
    # =========================================================================
    df /= df.loc[base_year]

    # =========================================================================
    # Labor Capital Intensity
    # =========================================================================
    df["capital_labor_ratio"] = df["capital"] / df["labor"]
    # =========================================================================
    # Labor Productivity
    # =========================================================================
    df["labor_productivity"] = df["product"] / df["labor"]
    # =========================================================================
    # Original: k=0.25, b=1.01
    # =========================================================================
    x = np.log(df["capital_labor_ratio"].astype(float))
    y = np.log(df["labor_productivity"].astype(float))

    alpha, intercept = np.polyfit(x, y, deg=1)
    scale = np.exp(intercept)

    # =========================================================================
    # Description
    # =========================================================================
    df["cap_to_lab"] = df["labor"] / df["capital"]
    # =========================================================================
    # Fixed Assets Turnover
    # =========================================================================
    df["capital_turnover"] = df["product"] / df["capital"]
    # =========================================================================
    # Product Trend Line=3 Year Moving Average
    # =========================================================================
    df["product_trend"] = df["product"].rolling(3, center=True).mean()
    df["product_gap"] = df["product"] - df["product_trend"]
    # =========================================================================
    # Computed Product
    # =========================================================================
    df["product_model"] = (
        scale * df["capital"] ** alpha * df["labor"] ** (1 - alpha)
    )
    print(df.columns)

    # =========================================================================
    # Computed Product Trend Line=3 Year Moving Average
    # =========================================================================
    df["product_model_trend"] = (
        df["product_model"].rolling(3, center=True).mean()
    )
    df["product_model_gap"] = df["product_model"] - df["product_model_trend"]
    # =========================================================================
    #     print(f"R**2: {r2_score(df["product"], df["product_model"]):,.4f}")
    #     print(df["product_model"].div(df["product"]).sub(1).abs().mean())
    # =========================================================================
    return df, alpha, scale


def labor_productivity_curve(
    lc_ratio: np.ndarray, alpha: float, scale: float
) -> np.ndarray:
    """P/L as a function of L/C."""
    return scale * lc_ratio ** (-alpha)


def capital_productivity_curve(
    lc_ratio: np.ndarray, alpha: float, scale: float
) -> np.ndarray:
    """P/C as a function of L/C."""
    return scale * lc_ratio ** (1 - alpha)


def plot_cobb_douglas(
    df: pd.DataFrame,
    alpha: float,
    scale: float,
    labels: dict[str, str],
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
    plt.title(labels["chart_inputs"].format(*df.index[[0, -1]], 1899))
    plt.grid()
    plt.legend()

    plt.figure(2)
    plt.semilogy(
        df.iloc[:, [2, 9]],
        label=[
            "Actual Product",
            "Computed Product, $P' = {:,.4f}L^{{{:,.4f}}}C^{{{:,.4f}}}$".format(
                scale,
                1 - alpha,
                alpha,
            ),
        ],
    )
    plt.xlabel("Period")
    plt.ylabel("Production")
    plt.title(labels["chart_actual_vs_model"].format(*df.index[[0, -1]], 1899))
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
    plt.title(labels["chart_gaps"])
    plt.grid()
    plt.legend()

    plt.figure(4)
    plt.plot(df.iloc[:, 9].div(df["product"]).sub(1))
    plt.xlabel("Period")
    plt.ylabel("Percentage Deviation")
    plt.title(labels["chart_relative_error"].format(*df.index[[0, -1]]))
    plt.grid()

    plt.figure(5, figsize=(5, 8))
    plt.scatter(df.iloc[:, 5], df.iloc[:, 4])
    plt.scatter(df.iloc[:, 5], df.iloc[:, 6])
    lc = np.arange(0.2, 1.0, 0.005)
    plt.plot(
        lc,
        labor_productivity_curve(lc, alpha, scale),
        label="$\\frac{3}{4}\\frac{P}{L}$",
    )
    plt.plot(
        lc,
        capital_productivity_curve(lc, alpha, scale),
        label="$\\frac{1}{4}\\frac{P}{C}$",
    )
    plt.xlabel("$\\frac{L}{C}$")
    plt.ylabel("Indexes")
    plt.title(labels["chart_productivities"])
    plt.grid()
    plt.legend()

    plt.show()


def figure_labels(base_year: int = 1899) -> dict[str, str]:
    return {
        "chart_inputs": f"Chart I Progress in Manufacturing {{}}$-${{}} ({base_year}=100)",
        "chart_actual_vs_model": f"Chart II Theoretical and Actual Curves of Production {{}}$-${{}} ({base_year}=100)",
        "chart_gaps": "Chart III Percentage Deviations of $P$ and $P'$ from Their Trend Lines\nTrend Lines=3 Year Moving Average",
        "chart_relative_error": "Chart IV Percentage Deviations of Computed from Actual Product {}$-${}",
        "chart_productivities": "Chart V Relative Final Productivities of Labor and Capital",
    }


def main() -> None:
    BASE_YEAR = 1899

    df = load_cobb_douglas_inputs().set_index("period")

    df, alpha, scale = estimate_cobb_douglas(df, base_year=BASE_YEAR)

    plot_cobb_douglas(
        df,
        alpha,
        scale,
        figure_labels(BASE_YEAR),
    )


if __name__ == "__main__":
    main()

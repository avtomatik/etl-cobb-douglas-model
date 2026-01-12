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


def estimate_cobb_douglas_ols(df: pd.DataFrame) -> tuple[float, float]:
    x = np.log(df["labor_capital_intensity"].astype(float))
    y = np.log(df["labor_productivity"].astype(float))

    alpha, intercept = np.polyfit(x, y, deg=1)
    scale = np.exp(intercept)

    return alpha, scale


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
    # Compute ratios and productivities
    # =========================================================================
    df["labor_capital_intensity"] = df["capital"] / df["labor"]
    df["labor_productivity"] = df["product"] / df["labor"]

    # =========================================================================
    # Additional calculated metrics
    # =========================================================================
    df["capital_labor_ratio"] = df["labor"] / df["capital"]
    df["capital_turnover"] = df["product"] / df["capital"]
    df["product_trend"] = df["product"].rolling(3, center=True).mean()
    df["product_gap"] = df["product"] - df["product_trend"]

    alpha, scale = estimate_cobb_douglas_ols(df)

    df["product_model"] = (
        scale * df["capital"] ** alpha * df["labor"] ** (1 - alpha)
    )
    df["product_model_trend"] = (
        df["product_model"].rolling(3, center=True).mean()
    )
    df["product_model_gap"] = df["product_model"] - df["product_model_trend"]
    df["product_model_error"] = df["product_model"] / df["product"] - 1

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


def figure_labels() -> dict[str, str]:
    return {
        "chart_inputs": "Chart I Progress in Manufacturing {start}$-${end} ({base_year}=100)",
        "chart_actual_vs_model": "Chart II Theoretical and Actual Curves of Production {start}$-${end} ({base_year}=100)",
        "chart_gaps": "Chart III Percentage Deviations of $P$ and $P'$ from Their Trend Lines\nTrend Lines=3 Year Moving Average",
        "chart_relative_error": "Chart IV Percentage Deviations of Computed from Actual Product {start}$-${end}",
        "chart_productivities": "Chart V Relative Final Productivities of Labor and Capital",
    }


def plot_cobb_douglas(
    df: pd.DataFrame,
    alpha: float,
    scale: float,
    labels: dict[str, str],
    base_year: int = 1899,
) -> None:
    """
    Cobb--Douglas Algorithm as per C.W. Cobb, P.H. Douglas. A Theory of Production, 1928;
    """
    assert df.shape[1] == 13, "Input df: pd.DataFrame shall have 13 columns."

    start, end = df.index[[0, -1]]

    formatted_labels = {
        key: value.format(start=start, end=end, base_year=base_year)
        for key, value in labels.items()
    }

    plt.figure(1)
    plt.semilogy(
        df[["capital", "labor", "product"]],
        label=["Fixed Capital", "Labor Force", "Physical Product"],
    )
    plt.xlabel("Period")
    plt.ylabel("Indexes")
    plt.title(formatted_labels["chart_inputs"])
    plt.grid()
    plt.legend()

    plt.figure(2)
    plt.semilogy(
        df[["product", "product_model"]],
        label=[
            "Actual Product",
            (
                f"Computed Product, "
                f"$P' = {scale:,.4f}L^{{{1 - alpha:,.4f}}}C^{{{alpha:,.4f}}}$"
            ),
        ],
    )
    plt.xlabel("Period")
    plt.ylabel("Production")
    plt.title(formatted_labels["chart_actual_vs_model"])
    plt.grid()
    plt.legend()

    plt.figure(3)
    plt.plot(
        df["product_gap"],
        label="Deviations of $P$",
        linestyle="-",
    )
    plt.plot(
        df["product_model_gap"],
        label="Deviations of $P'$",
        linestyle="--",
    )
    plt.xlabel("Period")
    plt.ylabel("Percentage Deviation")
    plt.title(formatted_labels["chart_gaps"])
    plt.grid()
    plt.legend()

    plt.figure(4)
    plt.plot(df["product_model_error"])
    plt.xlabel("Period")
    plt.ylabel("Percentage Deviation")
    plt.title(formatted_labels["chart_relative_error"])
    plt.grid()

    plt.figure(5, figsize=(5, 8))

    # =========================================================================
    # Observed productivities
    # =========================================================================
    plt.scatter(
        df["capital_labor_ratio"],
        df["labor_productivity"],
        alpha=0.7,
    )

    plt.scatter(
        df["capital_labor_ratio"],
        df["capital_turnover"],
        alpha=0.7,
    )

    # =========================================================================
    # Theoretical Cobb–Douglas curves
    # =========================================================================
    lc_grid = np.arange(0.2, 1.0, 0.005)

    plt.plot(
        lc_grid,
        labor_productivity_curve(lc_grid, alpha, scale),
        label=r"$\frac{3}{4} \frac{P}{L}$",
    )

    plt.plot(
        lc_grid,
        capital_productivity_curve(lc_grid, alpha, scale),
        label=r"$\frac{1}{4} \frac{P}{C}$",
    )

    # =========================================================================
    # Cosmetics
    # =========================================================================
    plt.xlabel(r"$\frac{L}{C}$")
    plt.ylabel("Indexes")
    plt.title(formatted_labels["chart_productivities"])
    plt.grid()
    plt.legend()

    plt.show()


def main() -> None:
    BASE_YEAR = 1899

    df = load_cobb_douglas_inputs().set_index("period")

    df, alpha, scale = estimate_cobb_douglas(df, base_year=BASE_YEAR)

    plot_cobb_douglas(
        df,
        alpha,
        scale,
        figure_labels(),
        BASE_YEAR,
    )


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def figure_labels() -> dict[str, str]:
    return {
        "chart_inputs": "Chart I Progress in Manufacturing {start}$-${end} ({base_year}=100)",
        "chart_actual_vs_model": "Chart II Theoretical and Actual Curves of Production {start}$-${end} ({base_year}=100)",
        "chart_gaps": "Chart III Percentage Deviations of $P$ and $P'$ from Their Trend Lines\nTrend Lines=3 Year Moving Average",
        "chart_relative_error": "Chart IV Percentage Deviations of Computed from Actual Product {start}$-${end}",
        "chart_productivities": "Chart V Relative Final Productivities of Labor and Capital",
    }


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
    labels: dict[str, str] | None = None,
) -> None:
    """
    Cobb--Douglas Algorithm as per C.W. Cobb, P.H. Douglas. A Theory of Production, 1928;
    """
    PLOT_CONTRACT_COLUMNS = {
        "capital_norm",
        "labor_norm",
        "product_norm",
        "labor_capital_intensity",
        "labor_productivity",
        "capital_labor_ratio",
        "capital_turnover",
        "product_trend",
        "product_gap",
        "product_model",
        "product_model_trend",
        "product_model_gap",
        "product_model_error",
    }

    missing = PLOT_CONTRACT_COLUMNS - set(df.columns)
    assert not missing, f"Plot contract violated. Missing columns: {missing}"

    if labels is None:
        labels = figure_labels()

    start, end = df["period"].iloc[[0, -1]]

    formatted_labels = {
        key: value.format(start=start, end=end, base_year=start)
        for key, value in labels.items()
    }

    plt.figure(1)
    plt.semilogy(
        df[["capital_norm", "labor_norm", "product_norm"]],
        label=["Fixed Capital", "Labor Force", "Physical Product"],
    )
    plt.xlabel("Period")
    plt.ylabel("Indexes")
    plt.title(formatted_labels["chart_inputs"])
    plt.grid()
    plt.legend()

    plt.figure(2)
    plt.semilogy(
        df[["product_norm", "product_model"]],
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

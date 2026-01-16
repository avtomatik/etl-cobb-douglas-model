import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz.model import capital_productivity_curve, labor_productivity_curve


def get_figure_labels() -> dict[str, str]:
    """
    Returns the labels for the various charts in the plot.
    """
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
    labels: dict[str, str] | None = None,
) -> None:
    """
    Generate plots based on the Cobb-Douglas production function and observed data.

    Parameters:
    df (pd.DataFrame): Dataframe containing the observed data
    alpha (float): Cobb-Douglas exponent for labor
    scale (float): Scaling factor
    labels (dict, optional): Custom chart labels
    """

    if labels is None:
        labels = get_figure_labels()

    start, end = df["period"].iloc[[0, -1]]

    formatted_labels = {
        key: value.format(start=start, end=end, base_year=start)
        for key, value in labels.items()
    }

    # Create the first figure: Inputs over time
    fig, ax = plt.subplots()
    ax.semilogy(df[["capital_norm", "labor_norm", "product_norm"]])
    ax.set_xlabel("Period")
    ax.set_ylabel("Indexes")
    ax.set_title(formatted_labels["chart_inputs"])
    ax.grid(True)
    ax.legend(["Fixed Capital", "Labor Force", "Physical Product"])

    # Create the second figure: Actual vs Model Production
    fig, ax = plt.subplots()
    ax.semilogy(df[["product_norm", "product_model"]])
    ax.set_xlabel("Period")
    ax.set_ylabel("Production")
    ax.set_title(formatted_labels["chart_actual_vs_model"])
    ax.grid(True)
    ax.legend(
        [
            "Actual Product",
            f"Computed Product, $P' = {scale:,.4f}L^{{{1 - alpha:,.4f}}}C^{{{alpha:,.4f}}}$",
        ]
    )

    # Create the third figure: Gaps in Product vs Model
    fig, ax = plt.subplots()
    ax.plot(df["product_gap"], label="Deviations of $P$", linestyle="-")
    ax.plot(
        df["product_model_gap"], label="Deviations of $P'$", linestyle="--"
    )
    ax.set_xlabel("Period")
    ax.set_ylabel("Percentage Deviation")
    ax.set_title(formatted_labels["chart_gaps"])
    ax.grid(True)
    ax.legend()

    # Create the fourth figure: Relative error in product model
    fig, ax = plt.subplots()
    ax.plot(df["product_model_error"])
    ax.set_xlabel("Period")
    ax.set_ylabel("Percentage Deviation")
    ax.set_title(formatted_labels["chart_relative_error"])
    ax.grid(True)

    # Create the fifth figure: Productivities and Cobb-Douglas curves
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.scatter(
        df["capital_labor_ratio"],
        df["labor_productivity"],
        alpha=0.7,
        label="Labor Productivity",
    )
    ax.scatter(
        df["capital_labor_ratio"],
        df["capital_turnover"],
        alpha=0.7,
        label="Capital Turnover",
    )

    lc_grid = np.arange(0.2, 1.0, 0.005)
    ax.plot(
        lc_grid,
        labor_productivity_curve(lc_grid, alpha, scale),
        label=r"$\frac{3}{4} \frac{P}{L}$",
    )
    ax.plot(
        lc_grid,
        capital_productivity_curve(lc_grid, alpha, scale),
        label=r"$\frac{1}{4} \frac{P}{C}$",
    )

    ax.set_xlabel(r"$\frac{L}{C}$")
    ax.set_ylabel("Indexes")
    ax.set_title(formatted_labels["chart_productivities"])
    ax.grid(True)
    ax.legend()

    plt.show()

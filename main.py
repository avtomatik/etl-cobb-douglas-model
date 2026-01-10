import io
from dataclasses import dataclass
from enum import Enum
from functools import cache
from http import HTTPStatus
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"


class Dataset(str, Enum):

    def __new__(cls, value: str, usecols: range):
        obj = str.__new__(cls)
        obj._value_ = value
        obj.usecols = usecols
        return obj

    DOUGLAS = "dataset_douglas.zip", range(4, 7)
    USA_BROWN = "dataset_usa_brown.zip", range(5, 8)
    USA_COBB_DOUGLAS = "dataset_usa_cobb-douglas.zip", range(5, 8)
    USA_KENDRICK = "dataset_usa_kendrick.zip", range(4, 7)
    USA_MC_CONNELL = "dataset_usa_mc_connell_brue.zip", range(1, 4)
    USCB = "dataset_uscb.zip", range(9, 12)

    def get_kwargs(self) -> dict[str, Any]:

        NAMES = ["series_id", "period", "value"]

        return {
            "filepath_or_buffer": DATA_DIR / self.value,
            "header": 0,
            "names": NAMES,
            "index_col": 1,
            "skiprows": (0, 4)[self.name in ["USA_BROWN"]],
            "usecols": self.usecols,
        }


class URL(Enum):
    FIAS = (
        "https://apps.bea.gov/national/FixedAssets/Release/TXT/FixedAssets.txt"
    )
    NIPA = "https://apps.bea.gov/national/Release/TXT/NipaDataA.txt"

    def get_kwargs(self) -> dict[str, Any]:

        NAMES = ["series_ids", "period", "value"]

        kwargs = {
            "header": 0,
            "names": NAMES,
            "index_col": 1,
            "thousands": ",",
        }
        if requests.head(self.value).status_code == HTTPStatus.OK:
            kwargs["filepath_or_buffer"] = io.BytesIO(
                requests.get(self.value).content
            )
        else:
            kwargs["filepath_or_buffer"] = self.value.split("/")[-1]
        return kwargs


@dataclass(frozen=True, eq=True)
class SeriesID:
    series_id: str
    source: Dataset | URL


@cache
def read_source(series_id: SeriesID) -> pd.DataFrame:
    """


    Parameters
    ----------
    series_id : SeriesID
        DESCRIPTION.

    Returns
    -------
    DataFrame
        ================== =================================
        df.index           Period
        df.iloc[:, 0]      Series IDs
        df.iloc[:, 1]      Values
        ================== =================================.

    """
    return pd.read_csv(**series_id.source.get_kwargs())


def pull_by_series_id(df: pd.DataFrame, series_id: SeriesID) -> pd.DataFrame:
    """


    Parameters
    ----------
    df : DataFrame
        ================== =================================
        df.index           Period
        df.iloc[:, 0]      Series IDs
        df.iloc[:, 1]      Values
        ================== =================================.
    series_id : SeriesID
        DESCRIPTION.

    Returns
    -------
    DataFrame
        ================== =================================
        df.index           Period
        df.iloc[:, 0]      Series
        ================== =================================.

    """
    assert df.shape[1] == 2
    return (
        df[df.iloc[:, 0] == series_id.series_id]
        .iloc[:, [1]]
        .rename(columns={"value": series_id.series_id})
    )


def stockpile(series_ids: list[SeriesID]) -> pd.DataFrame:
    """


    Parameters
    ----------
    series_ids : list[SeriesID]
        DESCRIPTION.

    Returns
    -------
    DataFrame
        ================== =================================
        df.index           Period
        ...                ...
        df.iloc[:, -1]     Values
        ================== =================================.

    """
    return pd.concat(
        map(lambda _: read_source(_).pipe(pull_by_series_id, _), series_ids),
        axis=1,
        sort=True,
    )


def combine_cobb_douglas(series_number: int = 3) -> pd.DataFrame:
    """
    Original Cobb--Douglas Data Collection Extension
    Parameters
    ----------
    series_number : int, optional
        DESCRIPTION. The default is 3.
    Returns
    -------
    DataFrame
        ================== =================================
        df.index           Period
        df.iloc[:, 0]      Capital
        df.iloc[:, 1]      Labor
        df.iloc[:, 2]      Product
        ================== =================================
    """
    MAP = {
        "CDT2S4": "capital",
        "CDT3S1": "labor",
        "J0014": "product",
        "J0013": "product_nber",
        "DT24AS01": "product_rev",
    }
    SERIES_IDS = [
        # =====================================================================
        # C.W. Cobb, P.H. Douglas Capital Series: Total Fixed Capital in 1880 dollars (4)
        # =====================================================================
        SeriesID("CDT2S4", Dataset.USA_COBB_DOUGLAS),
        # =====================================================================
        # C.W. Cobb, P.H. Douglas Labor Series: Average Number Employed (in thousands)
        # =====================================================================
        SeriesID("CDT3S1", Dataset.USA_COBB_DOUGLAS),
        # =====================================================================
        # Bureau of the Census, 1949, Page 179, J14: Warren M. Persons, Index of Physical Production of Manufacturing
        # =====================================================================
        SeriesID("J0014", Dataset.USCB),
        # =====================================================================
        # Bureau of the Census, 1949, Page 179, J13: National Bureau of Economic Research Index of Physical Output, All Manufacturing Industries.
        # =====================================================================
        SeriesID("J0013", Dataset.USCB),
        # =====================================================================
        # The Revised Index of Physical Production for All Manufacturing In the United States, 1899--1926
        # =====================================================================
        SeriesID("DT24AS01", Dataset.DOUGLAS),
    ]
    return (
        stockpile(SERIES_IDS)
        .rename(columns=MAP)
        .iloc[:, range(series_number)]
        .dropna(axis=0)
    )


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
    df = (
        combine_cobb_douglas()
        .pipe(transform_cobb_douglas, year_base=YEAR_BASE)[0]
        .iloc[:, range(5)]
    )

    _df, _params = df.pipe(transform_cobb_douglas, year_base=YEAR_BASE)
    plot_cobb_douglas(_df, _params, get_fig_map(YEAR_BASE))


if __name__ == "__main__":
    main()

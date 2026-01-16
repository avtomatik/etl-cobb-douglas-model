import numpy as np


def labor_productivity_curve(
    lc_ratio: np.ndarray, alpha: float, scale: float
) -> np.ndarray:
    """
    Compute labor productivity curve: P/L as a function of L/C.

    Parameters:
    lc_ratio (np.ndarray): Labor-to-capital ratio
    alpha (float): Cobb-Douglas exponent for labor
    scale (float): Scaling factor

    Returns:
    np.ndarray: Labor productivity as a function of the labor-to-capital ratio.
    """
    return scale * lc_ratio ** (-alpha)


def capital_productivity_curve(
    lc_ratio: np.ndarray, alpha: float, scale: float
) -> np.ndarray:
    """
    Compute capital productivity curve: P/C as a function of L/C.

    Parameters:
    lc_ratio (np.ndarray): Labor-to-capital ratio
    alpha (float): Cobb-Douglas exponent for labor
    scale (float): Scaling factor

    Returns:
    np.ndarray: Capital productivity as a function of the labor-to-capital ratio.
    """
    return scale * lc_ratio ** (1 - alpha)

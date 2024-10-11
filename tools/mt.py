""" Mathematical toolset used for ADEF framework. """


from __future__ import annotations

from typing import Tuple
import pandas as pd
import numpy as np
from scipy.stats import linregress


def interpolate_slope(
    t: np.ndarray, y: np.ndarray, step=1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method interpolates linearly the slope
    Args:
        t   - x axis values (start and stop)
        y   - y axis values (start and stop)
        step- step in which interpolation will be performed
    Returns:
        (<np.ndarray>, <np.ndarray>)  - list of new X interpolated values
                                        list of new Y interpolated values
    """
    d = linregress(t, y)
    t1 = np.arange(t[0], t[1], step=step)[1:]
    return t1, d.slope * t1 + d.intercept


def noise_gen(t: pd.Series, dt: int) -> np.ndarray:
    """
    Method that generates white and color noise
    Args:
        t: input data
        dt: iteration interval
    Returns:
        <np.ndarray>    - Noise that will be added to autoclave profile
    """
    # White noise
    nse = np.random.randn(t.shape[0])

    # Colored noise
    r = np.exp(-t / 0.5)
    cnse = np.convolve(nse, r, mode="same") * dt

    # Signal with a coherent part and a random part
    noise = 0.001 * np.sin(2 * np.pi * 10 * t) + cnse
    return noise

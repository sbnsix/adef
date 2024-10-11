""" Data noise filters module. """

import pandas as pd
import numpy as np

from scipy.fft import fft, ifft
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from pykalman import KalmanFilter
import padasip


class DataFilter:
    """
    Wrapper class for data filters that can be applied in data preprocessor element
    """

    @staticmethod
    def sr(l_data: pd.DataFrame) -> pd.Series:
        """
        Compute the Spectral Residual values of a time series.
        Args:
            l_data: input data for transformation
        Returns:
            <pd.Series> - output result with new set of
                          values representing spectral residual
        """
        if isinstance(l_data, pd.DataFrame):
            values = l_data[l_data.columns.tolist()[0]].to_numpy(dtype=float)
        elif isinstance(l_data, pd.Series):
            values = l_data.to_numpy(dtype=float)

        # Transform to log-spectrum
        eps = 1e-8  # to avoid log(0)
        log_spectrum = np.log(np.abs(fft(values)) + eps)

        # Calculate spectral residual
        avg_log_spectrum = np.convolve(log_spectrum, np.ones((3,)) / 3, mode="same")
        spectral_residual = log_spectrum - avg_log_spectrum

        # Transform back to time domain
        sr = np.abs(ifft(np.exp(spectral_residual)))

        # Apply Gaussian smoothing - to be decided
        # Add to DataFrame
        sr_val = sr.real
        sr_val[0] = sr_val[1]

        # Gaussian smoothing
        # Sigma is the standard deviation of the Gaussian kernel and controls the amount of smoothing
        sigma = 6

        # Apply Gaussian smoothing
        ssr = gaussian_filter1d(sr_val, sigma)  # .values

        return pd.Series(ssr)

    @staticmethod
    def sg(l_data: pd.Series, n_dimensions: int = 2) -> pd.Series:
        """
        Savitzky-Golay noise filter
        Args:
            l_data: input data
            n_dimensions: number of polynomial dimensions to consider when
                          constructing filtered data trace (careful with values
                          greater than 30 as it might cause some infeasibility problems).
        Returns:
            <pd.Series>
        """
        w = savgol_filter(l_data, l_data.shape[0], n_dimensions)
        return pd.Series(w)

    @staticmethod
    def kalman(
        l_data: pd.Series,
        transition_matrices: list,
        observation_matrices: list,
        initial_state_mean: float,
        initial_state_covariance: float,
        observation_covariance: float,
        transition_covariance: float,
    ) -> pd.Series:
        """
        Kalman noise filter
        Args:
            l_data: input
            transition_matrices: list,
            observation_matrices: list,
            initial_state_mean: float,
            initial_state_covariance: float,
            observation_covariance: float,
            transition_covariance: float
        Returns:
            <pd.Series>
        """
        kf = KalmanFilter(
            transition_matrices=transition_matrices,
            observation_matrices=observation_matrices,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
        )

        state_means, _ = kf.filter(l_data)
        state_means = state_means.flatten()

        return pd.Series(state_means)

    @staticmethod
    def lms(l_data: pd.Series) -> pd.Series:
        """
        Least Mean squares filter
        """
        result = padasip.filters.lms(l_data)

        return pd.Series(result)

    @staticmethod
    def derivative(data: pd.DataFrame, apply_mod: bool = False) -> pd.Series:
        """
        Signal derivative
        Args:
            data: input data
            apply_mod: apply absolute derivative - all samples > 0
        Returns:
            <pd.Series> - derivative for single column of data
        """
        dt = data.loc[:, ["time", "temp"]].copy(deep=False)

        dt.set_index("time", inplace=True)

        derv = dt.loc[:, "temp"].diff()
        if apply_mod:
            derv = abs(derv)
        derv.replace(np.nan, 0.0, inplace=True)

        return derv

    @staticmethod
    def mean_std(data: pd.Series, label_names: list) -> np.ndarray:
        """
        Mean/Standard deviation error data normalization
        Args:
            data: input data that will be used
            label_names: columns that will be looked at
                         during manifold process
        Returns:
            <np.ndarray> - array containing manifolded data
        """
        return (data - data.mean()) / data.std()

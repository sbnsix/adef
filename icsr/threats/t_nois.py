""" Complex attack - Noise signal added to normal trace """

from __future__ import annotations
import os
import sys
import pandas as pd
from threat_template import ThreatTemplate
import numpy as np
import random
from random import gauss

PATHS = ["./", "../tools"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from tools.log import Log
from mt import noise_gen
from threat_class import ThreatClass
from threat_tools import ThreatTools


class T_NOIS(ThreatTemplate):
    def __init__(self, cfg: dict, logger: Log) -> None:
        """
        CTOR
        Args:
            cfg: JSON dict containing threat configuration
            logger: Logger object
        Returns:
             <None>
        """
        super().__init__(cfg, logger)

    def noise(self, x: pd.Series, mu=0.0, std=0.15) -> pd.Series:
        """
        Method generates Gaussian noise
        Args:
            x: input time series based on which the noise will be generated
            mu: mu fidelity
            std: standard deviation
        Returns:
            <pd.Series> : series containing the noise of the
        """
        noise = np.random.normal(mu, std, size=x.shape)
        return noise

    def run(self, a_data: pd.DataFrame, start: int, stop: int) -> pd.DataFrame:
        """
        Method generates electronic attack on given data set
        Args:
            a_data: pandas data frame containing ground
                  truth data (original data set)
            start: start index for the attack inside data trace
            stop: stop of the attack inside data trace
        Returns:
            <pd.DataFrame>  - data frame with added attack
        """
        orig_data = a_data.copy(deep=True)

        # Add more noise to trace
        factor = random.randint(*self.cfg["sets"][0]["multipl"])

        y = self.noise(a_data.loc[start:stop, "tc"]) * factor

        a_data.loc[start:stop, "tc"] += y

        # Mark threat
        ThreatTools.mark_ground_truth(orig_data,
                                      a_data,
                                      "tc",
                                      self.cfg,
                                      start,
                                      stop,
                                      ThreatClass.NOIS)

        return a_data

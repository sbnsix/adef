""" Simple attack - high noise and then flat signal - sensor attack emulation """

from __future__ import annotations
import os
import sys
from random import gauss
import pandas as pd
from threat_template import ThreatTemplate
import random

PATHS = ["./", "../tools"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from mt import noise_gen
from tools.log import Log
from threat_class import ThreatClass
from threat_tools import ThreatTools


class T_HNFL(ThreatTemplate):
    """
    Implementation of high noise and low noise attack
    """

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

    def run(self, a_data: pd.DataFrame, start: int, stop: int) -> pd.DataFrame:
        """
        Method generates high and low noise attack on given data set
        Args:
            a_data: pandas data frame containing ground
                  truth data (original data set)
            start: start index for the attack inside data trace
            stop: stop of the attack inside data trace
        Returns:
            <pd.DataFrame>  - data frame with added attack
        """
        orig_data = a_data.copy(deep=True)
        avg_value = a_data.loc[start:stop, "tc"].mean()

        # Decide randomly how to split attack between high noise part
        # and flat part
        at_len = a_data.loc[start:stop, :].shape[0]
        mid_pt = start + int(at_len / random.randint(2, 5))

        # Add high noise
        high_noise_len = a_data.loc[start:mid_pt, "tc"].shape[0]
        y = pd.Series(
            [gauss(0.0, 1.0) for i in range(high_noise_len)]
        ) * random.randint(5, 15)
        a_data.loc[start:mid_pt, "tc"] += y.values

        # Flat noisy trace
        y = noise_gen(a_data.loc[mid_pt + 1: stop, "tc"], 0.5)

        if abs(avg_value - a_data.loc[mid_pt, "tc"]) > 2:
            a_data.loc[mid_pt + 1: stop, "tc"] = a_data.loc[mid_pt, "tc"]
        else:
            vrls = [avg_value - avg_value * 0.3, avg_value + avg_value * 0.3]
            x = (avg_value - a_data.loc[mid_pt, "tc"]) * 5
            flat_value = x if vrls[0] < x < vrls[1] else random.uniform(vrls[0], vrls[1])

            a_data.loc[mid_pt + 1: stop, "tc"] = flat_value

        a_data.loc[mid_pt + 1: stop, "tc"] += y.values

        ThreatTools.mark_ground_truth(orig_data,
                                      a_data,
                                      "tc",
                                      self.cfg,
                                      start,
                                      stop,
                                      ThreatClass.HNFL)

        return a_data

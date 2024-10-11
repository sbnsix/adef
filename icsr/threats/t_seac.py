""" Simple attack - sensor active control """

from __future__ import annotations
import os
import sys
import uuid
import random
import pandas as pd
import numpy as np
from scipy.stats import linregress
from threat_template import ThreatTemplate
from threat_tools import ThreatTools

PATHS = ["./", "../tools"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from tools.log import Log
from threat_class import ThreatClass
from threat_tools import ThreatTools


class T_SEAC(ThreatTemplate):
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
        random.seed(str(uuid.uuid4()))

    def run(self, a_data: pd.DataFrame, start: int, stop: int, cycle: int, sf_cfg: dict) -> pd.DataFrame:
        """
        Method generates electronic attack on given data set
        Args:
            a_data: pandas data frame containing ground
                  truth data (original data set)
            start: start index for the attack inside data trace
            stop: stop of the attack inside data trace
            cycle: cycle index inside the data trace
            sf_cfg: soft filter configuration
        Returns:
            <pd.DataFrame>  - data frame with added attack
        """
        orig_data = a_data.copy(deep=True)

        # Generate vector for attack
        cfg_set = self.cfg["sets"][0]
        val = random.randint(cfg_set["val"][0], cfg_set["val"][1])

        t = [start, stop + 1]
        y = [a_data.loc[start + 3, "temp"], val]
        d = linregress(t, y)
        t1 = np.arange(t[0], t[1], step=1)[1:]

        # self.log.debug(f"Shape: {str(a_data.shape)}, start {i + start}, stop {i + stop - 1}")
        a_data.loc[start : stop - 1, "tc"] = d.slope * t1 + d.intercept
        # a_data.loc[start: stop - 1, "label"] = 1

        a_data.loc[start - 1 : stop + 1] = ThreatTools.soft_filter(self.cfg["name"],
                                                                   a_data.loc[start - 1 : stop + 1],
                                                                   sf_cfg)

        # Mark threat
        ThreatTools.mark_ground_truth(orig_data,
                                      a_data,
                                      "tc",
                                      self.cfg,
                                      start,
                                      stop,
                                      ThreatClass.SEAC)

        return a_data

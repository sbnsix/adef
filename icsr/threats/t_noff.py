""" Complex attack - Noise and offset added to normal trace """

from __future__ import annotations
import os
import sys
import pandas as pd
from threat_template import ThreatTemplate
import random

PATHS = ["./", "../tools"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from tools.log import Log
from mt import interpolate_slope
from mt import noise_gen
from threat_class import ThreatClass
from threat_tools import ThreatTools


class T_NOFF(ThreatTemplate):
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

        # Noise part
        factor = random.randint(*self.cfg["sets"][0]["multipl"])
        a_data.loc[start:stop, "tc"] += noise_gen(a_data.loc[:, "tc"], 0.5) * factor

        # Offset part
        coff = random.randint(*self.cfg["sets"][0]["val"])
        a_data.loc[start:stop, "tc"] += coff

        # Attack length
        attack_l = int((stop - start) / 6)

        x = a_data.loc[start - attack_l : start + attack_l, "time"].index.tolist()
        y = a_data.loc[start - attack_l : start + attack_l, "tc"].tolist()

        # Add two linearize slops
        x1, y1 = interpolate_slope([x[0], x[-1]], [y[0], y[-1]])
        a_data.loc[start - attack_l + 1 : start + attack_l - 1, "tc"] = y1

        x = a_data.loc[stop - attack_l : stop + attack_l, "time"].index.tolist()
        y = a_data.loc[stop - attack_l : stop + attack_l, "tc"].tolist()
        x2, y2 = interpolate_slope([x[0], x[-1]], [y[0], y[-1]])

        a_data.loc[stop - attack_l + 1 : stop + attack_l - 1, "tc"] = y2

        # Mark threat
        ThreatTools.mark_ground_truth(orig_data,
                                      a_data,
                                      "tc",
                                      self.cfg,
                                      start - attack_l,
                                      stop + attack_l,
                                      ThreatClass.NOFF)
        return a_data

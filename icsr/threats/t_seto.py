""" Simple attack - SEnsor Temporary Offline """

from __future__ import annotations
import os
import sys
import uuid
import random
import pandas as pd

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


class T_SETO(ThreatTemplate):
    """
    Class describing sensor attack with temporary offline
    attack where sensor sends same value for attack period.
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
        random.seed(str(uuid.uuid4()))

    def run(self,
            a_data: pd.DataFrame,
            start: int,
            stop: int,
            cycle: int,
            sf_cfg: dict) -> pd.DataFrame:
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
        val = random.randint(*cfg_set["val"])

        a_data.loc[start:stop, "tc"] = val

        a_data.loc[start - 1 : stop + 1] = ThreatTools.soft_filter(self.cfg["name"],
                                                                   a_data.loc[start - 1: stop + 1],
                                                                   sf_cfg)

        # Mark threat
        ThreatTools.mark_ground_truth(orig_data,
                                      a_data,
                                      "tc",
                                      self.cfg,
                                      start,
                                      stop,
                                      ThreatClass.SETO)

        return a_data

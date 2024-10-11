""" Threat template class defining all common threat features"""

import os
import sys
import pandas as pd

PATHS = ["./", "../tools"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from log import Log


class ThreatTemplate:
    """
    Class describing generic threat behaviour
    """

    def __init__(self, cfg: dict, logger: Log) -> None:
        """
        CTOR
        Args:
            cfg - dictionary containing attack configuration
            logger - logger object
        Returns:
            <None>
        """
        self.cfg = cfg
        self.log = logger

    def run(
        self, data: pd.DataFrame, start: int, stop: int, cycle: int
    ) -> pd.DataFrame:
        """
        Method runs attack scenario in accordance to
        Args:
            data: input process data
            start: start index for the attack inside data trace
            stop: stop of the attack inside data trace
            cycle: cycle index inside the data trace
        Returns:
            <pd.DataFrame>  - data frame containing attacked data
        """
        raise NotImplementedError(
            "Please override threat run method for specific attack"
        )

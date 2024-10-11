""" Complex attack - emulation of Programmable Link Controller noise """

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from threat_template import ThreatTemplate
import random
import math

PATHS = ["./", "../tools"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from tools.log import Log
from mt import noise_gen
from mt import interpolate_slope
from threat_class import ThreatClass
from threat_tools import ThreatTools


class T_PLCN(ThreatTemplate):
    """
    Class modeling PLC(Programmable Link Controller) noise attack
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

    def normalise(self, x, ampl):
        max_int16 = ampl
        maxamp = max(x)
        amp = math.floor(max_int16 / maxamp)
        norm = np.zeros(len(x))
        for i in range(len(x)):
            norm[i] = amp * x[i]
        return norm

    def sine_plc(self, dur: int) -> np.ndarray:
        """
        Method generates exponentially attenuated sinusoid
        Args:
            dur: duration in discrete trace points
        Returns:
            <np.ndarray>    - arrays representing exponentially attenuated sinusoid
                              simulating PLC noise
        """
        # Approximate zero value where function has zero value
        zero_x = 6
        t = np.arange(zero_x, dur + zero_x, 1)
        sinusoid = 20 * ((np.sin(1 / 3 * t)) * (-np.exp(5 / t - 1))) # 1 / 2 *
        return sinusoid

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

        # Add PLC actuation to the trace
        a_data.loc[start : stop - 1, "tc"] += self.sine_plc(stop - start)

        # Add color noise after PLC actuation is completed
        factor = random.randint(*self.cfg["sets"][0]["multipl"])
        a_data.loc[start:stop, "tc"] += noise_gen(a_data.loc[:, "tc"], 0.5) * factor

        # Perform linear approximation at the end of the attack to make sure
        # smooth transition between applied attack and autoclave trace.
        a_data.loc[stop - 4 : stop + 4, "tc"] = interpolate_slope(
            [stop - 5, stop + 5],
            [a_data.loc[stop - 5, "tc"], a_data.loc[stop + 5, "tc"]],
        )[1]

        # Mark threat
        ThreatTools.mark_ground_truth(orig_data,
                                      a_data,
                                      "tc",
                                      self.cfg,
                                      start - 4,
                                      stop + 4,
                                      ThreatClass.PLCN)

        return a_data

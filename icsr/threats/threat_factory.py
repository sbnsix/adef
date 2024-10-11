""" Threat factory that generates attacks on the data set (cycle) """

import os
import sys
import pandas as pd
from threat_class import ThreatClass


PATHS = ["./", "../tools"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from log import Log
from auto_loader import AutoLoader


class ThreatFactory:
    threats = {}

    @staticmethod
    def init(cfg: dict, logger: object):
        if {} == ThreatFactory.threats:
            ThreatFactory.threats = AutoLoader.load(
                "./icsr/threats", "t_*.py", [cfg, logger]
            )

        if 0 == len(ThreatClass.values.keys()):
            cnt = 1
            for key in ThreatFactory.threats.keys():
                ThreatClass.add(key.upper(), cnt)
                cnt += 1

    @staticmethod
    def run(
        threat_id: ThreatClass,
        cfg: dict,
        data: pd.DataFrame,
        start: int,
        stop: int,
        cycle: int,
        sf_cfg: dict,
        logger: Log,
    ) -> pd.DataFrame:
        """
        Method runs threat factory
        Args:
            threat_id: threat class
            cfg: JSON representing threat configuration
            data: input data to be malformed during threat
                  activation process
            start: start attack index
            stop: stop attack index
            cycle: cycle index inside the trace
            sf_cfg: soft filter configuration
            logger: logger object
        Returns:
            <None>
        """
        # Map threat_id to its string representation
        threat_id_str = list({k: v for (k, v) in ThreatClass.values.items()
                              if ThreatClass.values[k] == threat_id}.keys())[0]

        # Supply threat from auto-loaded threat list
        threat_generator = ThreatFactory.threats[threat_id_str.lower()]
        threat_generator.cfg = cfg

        if ThreatClass.SETO == threat_id or \
            ThreatClass.SEAC == threat_id or \
            ThreatClass.MIXED == threat_id:
            input_args = [data, start, stop, cycle, sf_cfg]
        elif ThreatClass.NOIS == threat_id or \
            ThreatClass.PLCN == threat_id or \
            ThreatClass.COFF == threat_id or \
            ThreatClass.HNFL == threat_id or \
            ThreatClass.NOFF == threat_id:
            input_args = [data, start, stop]
        else:
            raise NotImplementedError(f"Unknown threat type: {threat_id}")

        a_data = threat_generator.run(*input_args)

        return a_data

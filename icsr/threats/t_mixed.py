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
from threat_factory import ThreatFactory

class T_MIXED(ThreatTemplate):
    """
    Class describing mixed attack.
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

    @staticmethod
    def attack_routine(
        cfg: dict,
        data: pd.DataFrame,
        data_mir: pd.DataFrame,
        p_attacker: object,
        density: int,
        attack_list: list,
        log: Log,
    ) -> pd.DataFrame:
        """
        Args:
            cfg: attack configuration
            data: ICS process input data
            data_mir: ICS mirrored ideal trace in case of apriori proces knowledge
            p_attacker: attack class responsible for generating an attack
            density: attack probability on a given trace
            attack_list: list of attacks used in mixed mode attack type
            log: logger object
        Returns:
            <pd.DataFrame> - attacked cycles
        """
        a_data = p_attacker.attack(data, cfg, density, attack_list)

        if a_data is not None:
            a_data.to_csv(cfg["attack"]["file"])
            if len(attack_list) < 2:
                log.debug(f"Attack [{attack_list}]: {cfg['attack']['file']} => OK")
        else:
            log.error(f"Data NOT generated for: {str(attack_list).upper()}")
            return None

        return a_data

    @staticmethod
    def mixed_attack(
        enabled_attacks: dict,
        a_path: str,
        density: int,
        p_attacker: object,
        data: pd.DataFrame,
        data_mir: pd.DataFrame,
        cfg: dict,
        log: Log,
    ) -> pd.DataFrame:
        """
        Method generates attacks on ICS cycle
        Args:
            enabled_attacks - list of enabled attacks
            a_path - attack path where attacks files will be written
            density - probability of attack density
            p_attacker - attacker object
            data - input process data
            data_mir - mirrored data used with apriori knowledge
            cfg - configuration object
            log - logger object
        Returns:
            <None>
        """

        cfg["attack"]["file"] = f"{a_path}a_data_mixed_{int(density * 100):02d}.csv"

        a_data = T_MIXED.attack_routine(cfg,
                                        data,
                                        data_mir if data_mir is not None else None,
                                        p_attacker,
                                        density,
                                        list(enabled_attacks.keys()),
                                        log)

        log.debug(
            f"Attack Detection [{list(enabled_attacks.keys())}] {cfg['attack']['file']} => OK"
        )
        return a_data

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

        # Mark threat
        ThreatTools.mark_ground_truth(orig_data,
                                      a_data,
                                      "tc",
                                      self.cfg,
                                      start,
                                      stop,
                                      ThreatClass.MIXED)

        return a_data

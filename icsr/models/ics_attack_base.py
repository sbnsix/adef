""" Autoclave process attacker implementation """


from __future__ import annotations
import os
import sys

import random
import uuid
import pandas as pd

from tools.log import Log

PATHS = ["./detectors", "./tools", "./threats", "./ics"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)


class ICSAttackBase:
    """
    Class implementing ICS attack class behavior in ADEF simulator
    """

    def __init__(self, cfg: dict, logger: Log) -> None:
        """
        CTOR
        Args:
            cfg: configuration object for autoclave attack
            logger: logger object
        Returns:
            <None>
        """
        self.log = logger
        self.cfg = cfg
        random.seed(str(uuid.uuid4()))

    def attack(
        self, data: pd.DataFrame, cfg: dict, density: float, attack_type: object = None
    ) -> pd.DataFrame:
        """
        Method generates attacks on autoclave profile
        depending on the configuration
        Args:
            data: original data set
            cfg: attack configuration
            density: probability of an attack on the given cycle
            attack_type: attack type that will be used during generation process
                         otherwise the attack trace is not generated
                         None - All enabled random of attacks will be used
                         list - Supplied list of attacks will be used to generate different attacks
        Returns:
            <pd.DataFrame>  - data with an attack
        """
        raise NotImplementedError(
            "Please add corresponding implementation of the attack type"
        )

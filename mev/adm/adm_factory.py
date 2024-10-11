""" ICS autoclave profile detector factory. """


from __future__ import annotations

import tools.log as log
# from ad_generic import AdmGeneric
from adm_enum import DetType

from auto_loader import AutoLoader

class DetFactory:
    """
    Class implements factory pattern for
    AD based detectors for autoclave profile
    used in the experiment to enable simplified testing
    with various scenarios.
    """
    ad = {}

    @staticmethod
    def init(cfg: dict, logger: object) -> None:
        if {} == DetFactory.ad:
            DetFactory.ad = AutoLoader.load("./mev/adm", "det_*.py", [logger])

        if 0 == len(DetType.values.keys()):
            cnt = 1
            for key in DetFactory.ad.keys():
                DetType.add(key.upper(), cnt)
                cnt += 1

    @staticmethod
    def create(class_type: str, logger: log.Log) -> AdmGeneric:
        """
        Method creates given detector class
        Args:
            class_type: type of the anomaly detector model described in Enum type
            logger:     logger object
        Returns:
            <Det_Generic>   - specific class derived from Det_Generic class
        """
        if class_type.lower() in DetFactory.ad.keys():
            return DetFactory.ad[class_type.lower()]
        else:
            raise NotImplementedError(
                f"Unknown or not implemented detector class of type {str(class_type)}"
            )

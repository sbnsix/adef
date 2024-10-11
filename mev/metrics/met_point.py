from __future__ import annotations
import os
import sys

PATHS = ["../", "../tools", "../metrics"]
for path in PATHS:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if path not in sys.path:
        sys.path.insert(1, path)

from log import Log
from met_base import AnomalyDetectionMetricsBase


class AnomalyDetectionMetricsPerPoint(AnomalyDetectionMetricsBase):
    """
    Class defining anomaly detection metrics per ROC point
    for ADEF framework
    """

    def __init__(self, step_name: str, cfg: dict, logger: Log) -> None:
        """
        CTOR
        Args:
            step_name: name of the step
            cfg: metric configuration
            logger: logger object
        Returns:
            <None>
        """
        super().__init__(step_name, cfg, logger)
